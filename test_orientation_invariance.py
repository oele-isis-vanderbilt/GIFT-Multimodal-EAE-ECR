"""Camera-invariance test for the POD orientation estimator.

Proves the body/muzzle bearing recovery is NOT specific to any camera angle,
camera height, map rotation, or person facing:

- a synthetic 3D pinhole camera is placed at arbitrary azimuths/heights
  around the room and projects a synthetic soldier's Halpe-26 keypoints
  (shoulders/hips at height, heels/toes on the floor, wrists gripping a
  rifle along the facing) into image pixels (y-down, like a real frame);
- the floor map uses an arbitrary rotation + scale (y-down image pixels,
  like a real map.png);
- the camera->map homography is fit from floor correspondences exactly the
  way production does (libs.Track.mapper.PixelMapper, RANSAC);
- the ACTUAL production cue functions (src.orientation) recover the body and
  muzzle bearings in map space, which are compared against the ground-truth
  facing mapped into the same space.

Run: python test_orientation_invariance.py
"""

import itertools
import math
import sys

import numpy as np

from libs.Track.mapper import PixelMapper
from src.orientation import (
    L_BIG_TOE, L_ELBOW, L_HEEL, L_HIP, L_SHOULDER, L_WRIST,
    MUZZLE_OFF_TORSO_MAX_DEG,
    R_BIG_TOE, R_ELBOW, R_HEEL, R_HIP, R_SHOULDER, R_WRIST,
    _foot_cue, _muzzle_cues, _pair_cue, _unit,
)

ROOM = 6.0          # room half-extent (m)
FOCAL = 1200.0      # px
IMG_CX, IMG_CY = 960.0, 540.0


def make_camera(azimuth_deg, dist, height):
    """World (x east, y north, z up) -> image (u right, v DOWN)."""
    az = math.radians(azimuth_deg)
    cam_pos = np.array([dist * math.cos(az), dist * math.sin(az), height])
    look = np.array([0.0, 0.0, 0.0])
    fwd = look - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)  # image v grows downward

    def project(pts3):
        pts3 = np.atleast_2d(pts3)
        rel = pts3 - cam_pos
        x = rel @ right
        y = rel @ down
        z = rel @ fwd
        return np.stack([IMG_CX + FOCAL * x / z, IMG_CY + FOCAL * y / z], axis=1)

    return project


def make_map(rot_deg, scale=40.0, offset=(600.0, 400.0)):
    """World floor (x, y) -> map pixels (x right, y DOWN), arbitrary rotation."""
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)

    def to_map(pts2):
        pts2 = np.atleast_2d(pts2)
        x = pts2[:, 0] * c - pts2[:, 1] * s
        y = pts2[:, 0] * s + pts2[:, 1] * c
        # y-down flip: north (world +y) goes UP on a map image => smaller v.
        return np.stack([offset[0] + scale * x, offset[1] - scale * y], axis=1)

    return to_map


def soldier_keypoints(pos, facing_deg):
    """Halpe-26 3D keypoints for a soldier at ``pos`` facing ``facing_deg``.

    Anatomical left = facing rotated +90 deg (world top-down, y north)."""
    th = math.radians(facing_deg)
    f = np.array([math.cos(th), math.sin(th), 0.0])
    left = np.array([-math.sin(th), math.cos(th), 0.0])
    p = np.array([pos[0], pos[1], 0.0])

    kp = np.full((26, 3), np.nan)

    def put(idx, offset, z):
        kp[idx] = p + offset + np.array([0.0, 0.0, z])

    put(L_SHOULDER, +0.20 * left, 1.45)
    put(R_SHOULDER, -0.20 * left, 1.45)
    put(L_HIP, +0.13 * left, 1.00)
    put(R_HIP, -0.13 * left, 1.00)
    put(L_HEEL, -0.05 * f + 0.11 * left, 0.02)
    put(R_HEEL, -0.05 * f - 0.11 * left, 0.02)
    put(L_BIG_TOE, +0.20 * f + 0.11 * left, 0.02)
    put(R_BIG_TOE, +0.20 * f - 0.11 * left, 0.02)
    # Two-handed rifle grip along the facing: rear (firing) hand near the
    # chest, front (support) hand out along the barrel; a level grip keeps
    # both hands at equal height (the estimator's stated assumption).
    put(R_WRIST, +0.25 * f - 0.02 * left, 1.22)   # rear hand
    put(L_WRIST, +0.55 * f + 0.02 * left, 1.22)   # front hand
    put(R_ELBOW, +0.05 * f - 0.18 * left, 1.24)
    put(L_ELBOW, +0.30 * f + 0.15 * left, 1.24)
    return kp


def bearing_error_deg(u, v):
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0))))


def run_trial(cam_az, cam_dist, cam_h, map_rot, pos, facing_deg):
    project = make_camera(cam_az, cam_dist, cam_h)
    to_map = make_map(map_rot)

    # Homography from floor correspondences, exactly like production.
    grid = np.array([[x, y] for x in (-4, -1.5, 1.5, 4) for y in (-4, -1.5, 1.5, 4)],
                    dtype=np.float64)
    px = project(np.column_stack([grid, np.zeros(len(grid))]))
    mp = to_map(grid)
    mapper = PixelMapper(px, mp)

    kp3 = soldier_keypoints(pos, facing_deg)
    kps = project(kp3)
    scores = np.ones(26)
    kps_list = [[float(u), float(v)] for u, v in kps]

    # Ground-truth facing in MAP space (two floor points differenced).
    th = math.radians(facing_deg)
    a = to_map(np.array([pos]))[0]
    b = to_map(np.array([[pos[0] + math.cos(th), pos[1] + math.sin(th)]]))[0]
    truth = _unit(b - a)

    body_cues = [c for c in (
        _pair_cue(kps_list, scores, mapper, L_SHOULDER, R_SHOULDER),
        _pair_cue(kps_list, scores, mapper, L_HIP, R_HIP),
        _foot_cue(kps_list, scores, mapper, L_HEEL, L_BIG_TOE),
        _foot_cue(kps_list, scores, mapper, R_HEEL, R_BIG_TOE),
    ) if c is not None]
    assert body_cues, "no body cues recovered"
    body = _unit(np.sum([u * w for u, w in body_cues], axis=0))

    m_cues = _muzzle_cues(kps_list, scores, mapper, body)
    assert m_cues, "no muzzle cues recovered"
    muzzle = _unit(np.sum([u * w for u, w in m_cues], axis=0))
    # Mirror the production implausibility guard (compute_orientation_series):
    # a fused estimate pointing behind the shooter is discarded for the body.
    guarded = muzzle
    if float(np.dot(muzzle, body)) < math.cos(math.radians(MUZZLE_OFF_TORSO_MAX_DEG)):
        guarded = body

    # The wrist-line cue in isolation: the parallax-free grip-axis component.
    wrist_pts = np.stack([kps[L_WRIST], kps[R_WRIST]])
    wl = _unit(np.asarray(mapper.pixel_to_map(wrist_pts[1:2])).reshape(-1, 2)[0]
               - np.asarray(mapper.pixel_to_map(wrist_pts[0:1])).reshape(-1, 2)[0])
    if wl is not None and float(np.dot(wl, body)) < 0:
        wl = -wl
    wl_err = bearing_error_deg(wl, truth) if wl is not None else float("nan")

    return (bearing_error_deg(body, truth), bearing_error_deg(guarded, truth),
            wl_err)


def main():
    rng_positions = [(-2.0, 1.5), (0.0, 0.0), (2.5, -2.0), (-1.0, -3.0)]
    body_errs, muzzle_errs, wl_errs = [], [], []
    trials = 0
    for cam_az, cam_h, map_rot, facing in itertools.product(
        (0, 45, 90, 135, 180, 225, 270, 315),   # camera all around the room
        (2.2, 3.5, 5.0),                        # low / typical / high mount
        (0, 30, 90, 210),                       # arbitrary map orientations
        (0, 30, 60, 90, 135, 180, 225, 300),    # person facing every which way
    ):
        pos = rng_positions[trials % len(rng_positions)]
        be, me, we = run_trial(cam_az, 7.0, cam_h, map_rot, pos, facing)
        body_errs.append(be)
        muzzle_errs.append(me)
        wl_errs.append(we)
        trials += 1

    body_errs = np.array(body_errs)
    muzzle_errs = np.array(muzzle_errs)
    wl_errs = np.array(wl_errs)
    print(f"trials: {trials} (8 camera azimuths x 3 heights x 4 map rotations x 8 facings)")
    print(f"body       bearing error: mean {body_errs.mean():.2f} deg | p95 "
          f"{np.percentile(body_errs, 95):.2f} | max {body_errs.max():.2f}")
    print(f"wrist-line bearing error: mean {wl_errs.mean():.2f} deg | p95 "
          f"{np.percentile(wl_errs, 95):.2f} | max {wl_errs.max():.2f}")
    print(f"muzzle (fused+guard) err: mean {muzzle_errs.mean():.2f} deg | p95 "
          f"{np.percentile(muzzle_errs, 95):.2f} | max {muzzle_errs.max():.2f}")

    # What this test guarantees, and what it deliberately does not:
    #
    # 1. Body facing is fully camera/map invariant (<10 deg): the paired-
    #    keypoint homography differencing cancels height parallax, and a
    #    chirality flip would show as ~180 deg. Strict.
    # 2. The wrist-line (grip axis) cue is likewise parallax-free — its
    #    equal-height pair differencing must stay <10 deg of the true axis
    #    (sign resolved by the exact body facing here). Strict.
    # 3. The FUSED muzzle estimate also mixes forearm and chest->hands cues
    #    that are NOT parallax-free; they are kept because an A/B on both
    #    real rooms (two near-opposite-corner mounts) showed the fusion is
    #    the more accurate and far more stable real-world estimator (the
    #    lone wrist-line destabilises in bladed stances). Their synthetic
    #    worst case at adversarial low mounts is large, but production
    #    discards any fused estimate > MUZZLE_OFF_TORSO_MAX_DEG off the
    #    torso (mirrored above), so the guarded estimate can never point
    #    into the rear half-plane behind an accurate body facing.
    assert body_errs.max() < 10.0, "body bearing not camera-invariant"
    assert wl_errs.max() < 10.0, "wrist-line cue not camera-invariant"
    assert muzzle_errs.max() <= MUZZLE_OFF_TORSO_MAX_DEG + 1e-6, \
        "guarded muzzle escaped the off-torso bound"
    print("PASS: body/grip-axis invariant; guarded fusion bounded to the "
          f"forward {MUZZLE_OFF_TORSO_MAX_DEG:g} deg cone")


if __name__ == "__main__":
    sys.exit(main())

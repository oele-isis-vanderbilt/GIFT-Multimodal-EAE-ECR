import argparse
import logging

from src.processing_engine import ProcessingEngine


def getargs():
    parser = argparse.ArgumentParser(
        description="GIFT Multimodal External Assessment Engine - Developed by Vanderbilt University")
    parser.add_argument('vmeta', type=str,
                        help="Path to the vmeta file")
    parser.add_argument('-f', '--force_transcode', action='store_true',
                        help="force re-encoding of videos before processing. Can help to resolve issues of video format or corruption.")
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Displayed detailing logging messages.")
    parser.add_argument('-o', '--output_path', type=str, default="output/",
                        help="Path to store the output (Default: output/)")
    
    return parser.parse_args()


def main():
    ######################################################Change Begin#####################################################
    # Appending the Vmeta path into a paths list, and passing this into  engine.mt_initialize as it expects a list of paths.
    args = getargs()
    paths=[]
    paths.append(args.vmeta)
    log_level = logging.DEBUG if args.verbose else logging.ERROR
    logging.basicConfig(format='[%(levelname)s] %(asctime)s : %(message)s', level=log_level)

    engine = ProcessingEngine(args.force_transcode)
    status = engine.mt_initialize(None, paths, args.output_path)
    ######################################################Change End#####################################################

    # engine = ProcessingEngine("tests", "models/crowdhuman.pt", False, True)
    # status = engine.mt_initialize(None, "tests/001.vmeta.xml")

    print()
    print(status)
    print()
    for metric in engine.metrics:
        print(metric)

if __name__ == "__main__":
    main()

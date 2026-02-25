import argparse
from distutils.log import ERROR
import logging
from xmlrpc.server import SimpleXMLRPCServer
from src.processing_engine import ProcessingEngine


def getargs():
    parser = argparse.ArgumentParser(
        description="GIFT Multimodal External Assessment Engine - Developed by Vanderbilt University")
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help="Port to run the XMLRPC on (Default: 8000)")
    parser.add_argument('-f', '--force_transcode', action='store_true',
                        help="Force re-encoding of videos before processing. Can help to resolve issues of video format or corruption.")
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Displayed detailing logging messages.")
    return parser.parse_args()

def main():
    args = getargs()

    log_level = logging.DEBUG if args.verbose else logging.ERROR
    logging.basicConfig(format='[%(levelname)s] %(asctime)s : %(message)s', level=log_level)

    # Start the XML RPC server on local host at the port specified via command line args
    server = SimpleXMLRPCServer(("localhost", args.port))
    server.logRequests = 0
    #server.allow_none = 1

    # Instantiate the class used for this server instance:
    server.register_instance(ProcessingEngine(args.force_transcode))
    server.register_function(lambda astr: '_' + astr, '_string')
    server.serve_forever()


if __name__ == "__main__":
    main()
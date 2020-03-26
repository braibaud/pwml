from __future__ import absolute_import

import flask as fl
import flask_restful as flr
import flask_restful.reqparse as flr_r
import argparse as ap
import traceback as tb

from ..classifiers import hierarchical as hc


# Class holding static properties only
class Statics(object):

    # Globals
    g_arguments = None
    g_model = None
    g_app = None
    g_api = None
    g_globals = {}


# Mapping HTTP requests to class functions
class Hosting(flr.Resource):

    def __init__(self):

        self.reqparse = flr_r.RequestParser()

        for feature in Statics.g_model.input_features:
            self.reqparse.add_argument(
                feature.feature_name,
                type=str,
                location='json',
                required=True,
                help='"'+ feature.feature_name +'" is mandatory ({error_msg})')

        super(Hosting, self).__init__()

    def post(self):

        success = False
        error = None
        error_details = None
        result = None

        try:
            args = self.reqparse.parse_args()
            
            intput = {}

            for feature in Statics.g_model.input_features:
                input[feature.feature_name] = args['feature.feature_name']

            result = Statics.g_model.predict(
                input=input)

            success = True

        except Exception as e:
            error_details = tb.format_exc()
            error = e.__str__()

        return {
            'success': success,
            'error': error,
            'error_details': error_details,
            'result': result
        }


def configure_model():

    print('Loading model...')

    Statics.g_model = hc.HierarchicalClassifierModel.load_model(
        filepath=Statics.g_arguments.modelpath)


def configure_api():

    Statics.g_app = fl.Flask('apps')
    Statics.g_app.config['BUNDLE_ERRORS'] = True

    Statics.g_api = flr.Api(Statics.g_app)

    Statics.g_globals['relative_url'] = '/{0}/{1}'.format(
        Statics.g_model.model_name,
        Statics.g_model.experiment_name)

    Statics.g_api.add_resource(
        Hosting, 
        Statics.g_globals['relative_url'])

    print('Serving model from "{0}"'.format(
        Statics.g_globals['relative_url']))


if __name__ == '__main__':

    print('Parsing arguments...')

    parser = ap.ArgumentParser(
        description='Start web-service')

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        type=str,
        help='The address the web-service is listening on.')

    parser.add_argument(
        '--port',
        default=5000,
        type=int,
        help='The port the web-service is listening on')

    parser.add_argument(
        '--modelpath',
        default='',
        type=str,
        required=True,
        help='The path of the model file to load.')

    Statics.g_arguments = parser.parse_args()

    configure_model()
    configure_api()

    Statics.g_app.run(
        host=Statics.g_arguments.host,
        port=Statics.g_arguments.port,
        debug=False)

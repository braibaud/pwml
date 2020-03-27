from __future__ import absolute_import

import flask as fl
import flask_restful as flr
import flask_restful.reqparse as flr_r
import flask_cors as flc
import argparse as ap
import traceback as tb

import pwml as pw
from pwml.classifiers import hierarchical as hc


# Class holding static properties only
class Statics(object):

    # Globals
    g_arguments = None
    g_models = {}
    g_app = None
    g_cors = None
    g_api = None


# Mapping HTTP requests to class functions
class Info(flr.Resource):

    def __init__(self):
        super(Info, self).__init__()

    def get(self):
        return {
            'success': True,
            'version': pw.__version__,
            'status': 'ready'
        }


class Hosting(flr.Resource):

    def __init__(self, **kwargs):

        self.model = Statics.g_models[kwargs['model_path']]
        self.reqparse = flr_r.RequestParser()

        for feature in self.model.input_features:
            self.reqparse.add_argument(
                feature.feature_name,
                type=str,
                location='json',
                required=True,
                help='"'+ feature.feature_name +'" is mandatory ({error_msg})')

        super(Hosting, self).__init__(**kwargs)

    def post(self):

        success = False
        error = None
        error_details = None
        result = None

        try:
            args = self.reqparse.parse_args()
            
            data = {}

            for feature in self.model.input_features:
                data[feature.feature_name] = args[feature.feature_name]

            result = self.model.predict(
                input=data)

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


def configure_models():

    print('Loading models...')

    for model_path in Statics.g_arguments.modelpath:
        Statics.g_models[model_path] = hc.HierarchicalClassifierModel.load_model(
            filepath=model_path)


def configure_api():

    Statics.g_app = fl.Flask('apps')
    Statics.g_app.config['BUNDLE_ERRORS'] = True

    Statics.g_cors = flc.CORS(
        Statics.g_app, 
        resources={r'*': {'origins': '*'}})

    Statics.g_api = flr.Api(Statics.g_app)

    Statics.g_api.add_resource(
        Info, 
        '/info')

    for model_path, model in Statics.g_models.items():
        Statics.g_api.add_resource(
            resource=Hosting, 
            urls='/{0}/{1}'.format(
                model.model_name,
                model.experiment_name),
            endpoint='***{0}${1}***'.format(
                model.model_name,
                model.experiment_name),
            resource_class_kwargs={'model_path': model_path})

        print('Serving model "{0}" from "{1}"'.format(
            model_path,
            '/{0}/{1}'.format(
                model.model_name,
                model.experiment_name)))


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
        nargs='+',
        type=str,
        required=True,
        help='A list of paths of models to load.')

    Statics.g_arguments = parser.parse_args()

    configure_models()
    configure_api()

    Statics.g_app.run(
        host=Statics.g_arguments.host,
        port=Statics.g_arguments.port,
        debug=False)

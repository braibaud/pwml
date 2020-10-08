from __future__ import absolute_import

import flask as fl
import flask_restful as flr
import flask_restful.reqparse as flr_r
import flask_cors as flc
import flask_wtf.csrf as flwc

import argparse as ap
import traceback as tb
import os

# Comment next line to avoid disabling CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

# Specify TensorflowHub cache location (comment next line to use default location)
os.environ['TFHUB_CACHE_DIR'] = 'C:\\DEV\\TensorFlowCache'


# -----------------------------------------------------------------
# Example of batch file used to run the web-service (on port 5000):
#   * activates the conda environment
#   * start the web-service & load 2 models "v1/L15" and "v1/BRAND"
#   * deactivate the conda environment on termination
# -----------------------------------------------------------------
#
# call C:\ProgramData\Anaconda3\condabin\activate.bat exts-tf2-gpu
# python.exe C:\DEV\tf2\modelhosting.py --host 0.0.0.0 --port 5000 --models "v1/L15|C:\DEV\tf2\global-plan-hierarchy-fulltext_MODEL.p" "v1/BRAND|C:\DEV\tf2\global-plan-brand-fulltext_MODEL.p"
# call C:\ProgramData\Anaconda3\condabin\deactivate.bat
#
# -----------------------------------------------------------------


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

        print('__init__ for model "{0}"'.format(kwargs['model_id']))

        self.model = Statics.g_models[kwargs['model_id']]['model']
        self.reqparse = flr_r.RequestParser()

        for feature in self.model.input_features:
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

    for idx, text in enumerate(Statics.g_arguments.models):

        if '|' not in text:
            print('Model #{0} parameter is invalid (missing "|").'.format(idx))
            continue
    
        parts = text.split('|')

        if len(parts) != 2:
            print('Model #{0} parameter is invalid (invalid number of "|").'.format(idx))
            continue

        model_id = parts[0].strip().lower()
        model_path = parts[1]

        print('Model #{0} parameters:'.format(idx))
        print('  id: "{0}"'.format(model_id))
        print('  path: "{0}"'.format(model_path))

        if not os.path.exists(model_path):
            print('Model #{0} (id: "{1}") parameter is invalid (model file missing at "{2}").'.format(idx, model_id, model_path))
            continue

        if model_id in Statics.g_models:
            print('Model #{0} (id: "{1}") has already been defined.'.format(idx, model_id))
            continue

        Statics.g_models[model_id] = {
            'id': model_id,
            'path': model_path,
            'model': hc.HierarchicalClassifierModel.load_model(
                filepath=model_path)
        }


def configure_api():

    Statics.g_app = fl.Flask('apps')
    Statics.g_app.config['BUNDLE_ERRORS'] = True

    Statics.g_csrf = flwc.CSRFProtect()
    Statics.g_csrf.init_app(
        Statics.g_app)

    Statics.g_cors = flc.CORS(
        Statics.g_app, 
        resources={r'/*': {'origins': '*', 'send_wildcard': 'False'}})

    Statics.g_api = flr.Api(
        app=Statics.g_app,
        catch_all_404s=True)

    Statics.g_api.add_resource(
        Info, 
        '/info',
        endpoint='info')

    for _, model in Statics.g_models.items():

        url = '/api/{0}'.format(model['id'])
        endpoint = '{0}'.format(model['id'])

        Statics.g_api.add_resource(
            Hosting, 
            url,
            endpoint=endpoint,
            resource_class_kwargs={'model_id': model['id']})

        print('Serving model "{0}" from "{1}"'.format(
            model['id'],
            url))


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
        '--models',
        nargs='+',
        type=str,
        required=True,
        help='A list of "model-id|model-path" to load.')

    Statics.g_arguments = parser.parse_args()

    configure_models()
    configure_api()

    Statics.g_app.run(
        host=Statics.g_arguments.host,
        port=Statics.g_arguments.port,
        debug=True)

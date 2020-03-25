
class InputFeature(object):

    def __init__(self, feature_name, feature_type):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.classes = None


class OutputFeature(object):

    def __init__(self, feature_name, child_feature=None):
        self.feature_name = feature_name
        self.child_feature = child_feature

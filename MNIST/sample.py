import rasterization_tools
import json
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import predictor


class Sample:
    COUNT = 0

    def __init__(self, desc, label, seed):
        self.id = Sample.COUNT
        self.seed = seed
        self.features = {}
        self.xml_desc = desc
        self.purified = rasterization_tools.rasterize_in_memory(self.xml_desc)
        self.expected_label = label
        self.predicted_label = None
        self.confidence = None
        self.performance = None
        Sample.COUNT += 1

    def to_dict(self):
        return {'id': str(self.id),
                'seed': str(self.seed),
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'misbehaviour': self.is_misbehavior(),
                'performance': str(self.confidence),
                'features': self.features
    }

    def evaluate(self):
        ff = None          
        self.predicted_label, self.confidence = \
            predictor.Predictor.predict(self.purified)

        # Calculate fitness function
        ff = self.confidence if self.confidence > 0 else -0.1
            
        return ff

    def from_dict(self, the_dict):
        for k in self.__dict__.keys():
            if k in the_dict.keys():
                setattr(self, k, the_dict[k])
        return self

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_png(self, filename):
        plt.imsave(filename+'.png', self.purified.reshape(28, 28), cmap='gray', format='png')

    def save_npy(self, filename):
        np.save(filename, self.purified)
        test_img = np.load(filename+'.npy')
        diff = self.purified - test_img
        assert(np.linalg.norm(diff) == 0)

    def save_svg(self, filename):
        data = self.xml_desc
        filedest = filename + ".svg"
        with open(filedest, 'w') as f:
            f.write(data)

    def is_misbehavior(self):
        if str(self.expected_label) == str(self.predicted_label):
            return False
        else:
            return True

    def export(self, dst):
        dst = join(dst, "mbr"+str(self.id))
        self.dump(dst)
        self.save_npy(dst)
        self.save_png(dst)
        self.save_svg(dst)

    def clone(self):
        clone_digit = Sample(self.xml_desc, self.expected_label, self.seed)
        return clone_digit
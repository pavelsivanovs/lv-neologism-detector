import os.path
from subprocess import PIPE, Popen
from typing import List


class Lemmatizer:
    """ Encapsulation class for a LVTagger Morphotagging method
    implemented in Peterisp/LVTagger (https://github.com/PeterisP/LVTagger). """

    # FIXME: don't work for multiple inputs. can try `pexpect`:
    #  https://stackoverflow.com/questions/28616018/multiple-inputs-and-outputs-in-python-subprocess-communicate

    def __init__(self):
        morphotagger_script_path = os.path.abspath(os.path.join(os.path.curdir, 'tagger', 'morphotagger.sh'))
        command = ['bash', morphotagger_script_path, '-lemmatized-text']
        self.lemmatizer = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)

    def __del__(self):
        self.lemmatizer.kill()

    def lemmatize(self, text: str) -> List[str]:
        if text == '':
            raise Exception('Unintentional termination of the lemmatizer process by passing an empty string.')
        output, error = self.lemmatizer.communicate(text)

        # Commenting out because retrieved error is related to the LVTagger
        # if error:
        #     raise Exception(f'''
        #     Error occurred during lemmatization of text: {text}
        #     Retrieved output: {output}
        #     Error: {error}
        #     ''')

        return output.split()

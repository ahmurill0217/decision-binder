"""Small test doubles for formatter contracts, not a substitute d1 tokenizer."""
import json


class Tokenizer:
    bos_token = '<|startoftext|>'
    eos_token = '<|im_end|>'
    bos_token_id = 1000000
    eos_token_id = 1000001

    def encode(self, text, **kwargs):
        ids = []
        while text:
            for token, identifier in ((self.bos_token, self.bos_token_id), (self.eos_token, self.eos_token_id)):
                if text.startswith(token):
                    ids.append(identifier)
                    text = text[len(token):]
                    break
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids, **kwargs):
        special = {self.bos_token_id: self.bos_token, self.eos_token_id: self.eos_token}
        return ''.join(special[i] if i in special else chr(i) for i in ids)


class Utility:
    IM_START = '<|im_start|>'

    @staticmethod
    def as_question(question):
        return question

    @staticmethod
    def aliases(tokenizer, options):
        return [(code, ord(code)) for code in 'AB'[:len(options)]]

    @staticmethod
    def render(tokenizer, state, question, bos, **kwargs):
        options = '\n'.join(f'{code} {choice}' for code, choice in zip('AB', question['criteria']))
        return (bos + '<|im_start|>user\n' + json.dumps(state, indent=2)
                + '\n\n\nQUESTION:\n' + question['instructions'] + '\n\nOptions:\n'
                + options + '<|im_end|>\n<|im_start|>assistant\n')

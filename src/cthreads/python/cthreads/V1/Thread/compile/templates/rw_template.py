import os
import re

def write_template(template_name: str, output_file: str, **kwargs):
    template_path = os.path.join(os.path.dirname(__file__), template_name)
    try:
        with open(template_path, 'r') as f_in:
            template = f_in.read()
            double_braces = re.compile(r'{{{(.*?)}}}', re.DOTALL)

            rendered = double_braces.sub(lambda match: kwargs.get(match.group(1), ''), template)
            with open(output_file, 'w') as f_out:
                f_out.write(rendered)
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file {template_path} not found")
    except Exception as e:
        raise Exception(f"Error writing template {template_name} to {output_file}: {e}")
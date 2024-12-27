import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from scikitplot import __version__

context = {
  # "version": __version__.split('+')[0],
  "version": __version__,
}

# Load your template
output_path = os.path.join(os.getcwd(), '_static')
os.makedirs(output_path, exist_ok=True)
env = Environment(loader=FileSystemLoader('.'))
switcher_json = env.get_template('switcher.json.template')

def main():
  for template, f_out in zip([switcher_json,], ['switcher.json']):
    # Render the template with actual values
    output = template.render(context)
    
    # Print the rendered result or save it to a file
    with open(os.path.join(output_path, f_out), 'w') as output_file:
        output_file.write(output)  
    print(f"{os.path.join(output_path, f_out)} file created successfully.")

if __name__ == "__main__":
  main()
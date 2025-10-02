========================
CODE SNIPPETS
========================
TITLE: Complete Gradio Client CDN Example in HTML
DESCRIPTION: A full HTML example demonstrating how to include the Gradio Client via CDN and connect to a Gradio app ('abidlabs/en2fr') to perform a prediction. This illustrates a complete setup for browser-based usage.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_3

LANGUAGE: html
CODE:
```
<!DOCTYPE html>
<html lang="en">
<head>
    <script type="module">
        import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";
        const client = await Client.connect("abidlabs/en2fr");
        const result = await client.predict("/predict", {
            text: "My name is Hannah"
        });
        console.log(result);
    </script>
</head>
</html>
```

----------------------------------------

TITLE: Install Gradio Client via npm
DESCRIPTION: Instructions for installing the @gradio/client package using npm, suitable for Node.js (version >=18.0.0) and browser-based projects. This command adds the package to your project dependencies.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_1

LANGUAGE: bash
CODE:
```
npm i @gradio/client
```

----------------------------------------

TITLE: Connect to a Gradio App via Full URL
DESCRIPTION: This example shows how to connect to a Gradio application hosted on any server by providing its full URL. This is useful for apps not hosted on Hugging Face Spaces, such as those running on a share URL.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_5

LANGUAGE: python
CODE:
```
from gradio_client import Client

client = Client("https://bec81a83-5b5c-471e.gradio.live")
```

----------------------------------------

TITLE: Install Gradio Python Library
DESCRIPTION: Installs the Gradio library, a popular Python library for building machine learning web applications, using pip. The -q flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/test_chatinterface_examples/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Client Library
DESCRIPTION: This command installs or upgrades the `gradio_client` Python package using pip. It ensures you have the latest version of the library, which is compatible with Python 3.10 or higher.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_1

LANGUAGE: bash
CODE:
```
$ pip install --upgrade gradio_client
```

----------------------------------------

TITLE: Gradio Dropdown Component Demo Setup and Launch
DESCRIPTION: This example provides the complete Python code to set up and run a Gradio application featuring a basic dropdown component. It includes the necessary library installation step and the core application logic for defining and launching the interface.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/dropdown_component/run.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
!pip install -q gradio
```

LANGUAGE: Python
CODE:
```
import gradio as gr

with gr.Blocks() as demo:
    gr.Dropdown(choices=["First Choice", "Second Choice", "Third Choice"])

demo.launch()
```

----------------------------------------

TITLE: Start Gradio Website Development Server
DESCRIPTION: Launches the local development server for the Gradio website, enabling real-time preview of documentation changes for both API references and guides.

SOURCE: https://github.com/gradio-app/gradio/blob/main/CONTRIBUTING.md#_snippet_20

LANGUAGE: bash
CODE:
```
pnpm dev
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a prerequisite for running the Gradio application. The `-q` flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_zoom_sync/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a Python package installer. The `-q` flag ensures a quiet installation, suppressing most output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_aggregate_quantitative/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install or upgrade Gradio using pip
DESCRIPTION: This command installs or upgrades the Gradio Python package using pip, the standard package installer for Python. It is recommended to perform this installation within a virtual environment to manage dependencies effectively.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/01_getting-started/01_quickstart.md#_snippet_0

LANGUAGE: Bash
CODE:
```
pip install --upgrade gradio
```

----------------------------------------

TITLE: Gradio Demo with Session State Implementation
DESCRIPTION: Provides a server-side Gradio application example demonstrating how to use `gr.State` to maintain session-specific information. This demo tracks and counts user-submitted words across interactions within a single session.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_20

LANGUAGE: Python
CODE:
```
import gradio as gr

def count(word, list_of_words):
    return list_of_words.count(word), list_of_words + [word]

with gr.Blocks() as demo:
    words = gr.State([])
    textbox = gr.Textbox()
    number = gr.Number()
    textbox.submit(count, inputs=[textbox, words], outputs=[number, words])
    
demo.launch()
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio Python library using pip, a prerequisite for running the demo.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_series_nominal/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Create a basic Gradio 'Hello World' application
DESCRIPTION: This Python code defines a simple function `greet` that takes a name and an intensity level. It then creates a Gradio interface using `gr.Interface`, wrapping the `greet` function with a text input for the name, a slider input for intensity, and a text output for the greeting. The `demo.launch()` method starts the web application, making it accessible locally.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/01_getting-started/01_quickstart.md#_snippet_1

LANGUAGE: Python
CODE:
```
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(lines=2, placeholder="Name Here..."), gr.Slider(minimum=1, maximum=10, step=1, label="Intensity")],
    outputs="text",
)

if __name__ == "__main__":
    demo.launch()
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip. The `-q` flag ensures a quiet installation, suppressing verbose output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/hello_world/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, the Python package installer. The '-q' flag ensures a quiet installation process.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_aggregate_temporal/run.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Python Library
DESCRIPTION: Installs the Gradio Python library silently using pip, a common first step for setting up Gradio applications.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/markdown_example/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Connect to Public Hugging Face Space with Gradio Client
DESCRIPTION: Illustrates how to establish a connection to a public Gradio application hosted on Hugging Face Spaces using the Client.connect() method. This example connects to a translation Space.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_4

LANGUAGE: js
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.connect("abidlabs/en2fr"); // a Space that translates from English to French
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a prerequisite for running Gradio applications. The `-q` flag ensures a quiet installation, suppressing verbose output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/test_chatinterface_multimodal_examples/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Make a simple prediction with Gradio Client
DESCRIPTION: Demonstrates how to connect to a Gradio application and make a basic prediction using the `.predict()` method with a single parameter.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_12

LANGUAGE: JavaScript
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.connect("abidlabs/en2fr");
const result = await app.predict("/predict", ["Hello"]);
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, which is essential for running Gradio applications. The '-q' flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_series_quantitative/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library silently using pip, a prerequisite for running the Gradio application. This command ensures all necessary Gradio components are available.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/chatbot_examples/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Basic Usage of gradio.Examples
DESCRIPTION: Demonstrates the basic instantiation syntax for the `gradio.Examples` component within a Gradio application, showing how to initialize it with a list of example inputs.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/gradio/04_helpers/12_examples.svx#_snippet_0

LANGUAGE: python
CODE:
```
gradio.Examples(···)
```

----------------------------------------

TITLE: Prepare Example Data for Gradio Demo
DESCRIPTION: Sets up the environment by creating an 'examples' directory and downloads a sample CSV file ('log.csv') from the Gradio GitHub repository. This file is intended to be used as pre-filled examples within the Gradio interface.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/calculator/run.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
import os
os.mkdir('examples')
!wget -q -O examples/log.csv https://github.com/gradio-app/gradio/raw/main/demo/calculator/examples/log.csv
```

----------------------------------------

TITLE: Connect to Authenticated Gradio App with JavaScript
DESCRIPTION: This example demonstrates how to connect to a Gradio application that requires authentication. The username and password are provided as a tuple to the 'auth' argument of the Client.connect method.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_9

LANGUAGE: javascript
CODE:
```
import { Client } from "@gradio/client";

Client.connect(
  space_name,
  { auth: [username, password] }
)
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, which is necessary to run Gradio applications. The '-q' flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/hello_world_2/run.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio Python library, a prerequisite for running the interactive demo. The `-q` flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_zoom/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Start Svelte development server
DESCRIPTION: Commands to start the development server for a Svelte application after dependencies are installed. Includes an option to automatically open the application in a new browser tab.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/component-test/README.md#_snippet_1

LANGUAGE: bash
CODE:
```
npm run dev

npm run dev -- --open
```

----------------------------------------

TITLE: Create and Launch Gradio Image Flip Demo with Examples
DESCRIPTION: Defines a Gradio application using `gr.Blocks` that allows users to upload an image, flip it 180 degrees, and view the output. It integrates the `gr.Examples` component to provide pre-loaded image examples, demonstrating how to use local files to populate the input for quick testing and demonstration.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/examples_component/run.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
import gradio as gr
import os

def flip(i):
    return i.rotate(180)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            img_i = gr.Image(label="Input Image", type="pil")
        with gr.Column():
            img_o = gr.Image(label="Output Image")
    with gr.Row():
        btn = gr.Button(value="Flip Image")
    btn.click(flip, inputs=[img_i], outputs=[img_o])

    gr.Examples(
        [
            os.path.join(os.path.abspath(''), "images/cheetah1.jpg"),
            os.path.join(os.path.abspath(''), "images/lion.jpg")
        ],
        img_i,
        img_o,
        flip
    )

demo.launch()
```

----------------------------------------

TITLE: Connect to a Private Hugging Face Gradio Space
DESCRIPTION: This example demonstrates connecting to a private Gradio application hosted on Hugging Face Spaces. It requires passing your Hugging Face token via the `hf_token` parameter for authentication.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_3

LANGUAGE: python
CODE:
```
from gradio_client import Client

client = Client("abidlabs/my-private-space", hf_token="...")
```

----------------------------------------

TITLE: Download Example Images for Gradio Demo
DESCRIPTION: Creates an 'images' directory and downloads several image files (JPG, WEBP, PNG) from the Gradio GitHub repository. These images serve as pre-defined examples for the Gradio application's input.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/examples_component/run.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
import os
os.mkdir('images')
!wget -q -O images/cheetah1.jpg https://github.com/gradio-app/gradio/raw/main/demo/examples_component/images/cheetah1.jpg
!wget -q -O images/lion.jpg https://github.com/gradio-app/gradio/raw/main/demo/examples_component/images/lion.jpg
!wget -q -O images/lion.webp https://github.com/gradio-app/gradio/raw/main/demo/examples_component/images/lion.webp
!wget -q -O images/logo.png https://github.com/gradio-app/gradio/raw/main/demo/examples_component/images/logo.png
```

----------------------------------------

TITLE: Load Gradio Client via jsDelivr CDN (Module Import)
DESCRIPTION: Shows how to quickly load the @gradio/client library directly into an HTML file using a script tag with type="module" and the jsDelivr CDN. This method is ideal for prototyping.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_2

LANGUAGE: html
CODE:
```
<script type="module">
	import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";
	...
</script>
```

----------------------------------------

TITLE: Duplicate Gradio Space with Hardware Configuration in JavaScript
DESCRIPTION: This example shows how to duplicate a Gradio Space while specifying hardware and timeout options. This is useful for managing costs associated with GPU usage by setting a specific hardware tier and an inactivity timeout for the duplicated Space.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_7

LANGUAGE: javascript
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.duplicate("abidlabs/whisper", {
	hf_token: "hf_...",
	timeout: 60,
	hardware: "a10g-small"
});
```

----------------------------------------

TITLE: Add Examples to Gradio Interface for User Interaction
DESCRIPTION: Demonstrates how to add the "examples" parameter to gr.Interface to provide users with pre-defined input sets. The "examples" parameter expects a list of lists, where each inner list corresponds to the ordered inputs of the interface.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/10_other-tutorials/create-your-own-friends-with-a-gan.md#_snippet_6

LANGUAGE: python
CODE:
```
gr.Interface(
    # ...
    # keep everything as it is, and then add
    examples=[[123, 15], [42, 29], [456, 8], [1337, 35]],
).launch(cache_examples=True) # cache_examples is optional
```

----------------------------------------

TITLE: API Documentation for Gradio Session State Demo
DESCRIPTION: Documents the API endpoint for the Gradio demo with session state, as seen by the Python client. It shows the single input and output exposed, with the client handling internal state management automatically for subsequent requests.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_21

LANGUAGE: APIDOC
CODE:
```
Client.predict() Usage Info
---------------------------
Named API endpoints: 1

 - predict(word, api_name="/count") -> value_31
    Parameters:
     - [Textbox] word: str (required)  
    Returns:
     - [Number] value_31: float 
```

----------------------------------------

TITLE: GitHub Action: install-frontend-deps
DESCRIPTION: This GitHub Action is designed to set up the frontend development environment. It installs Node.js, pnpm, and then proceeds to build the frontend components of the Gradio project. It's a reusable action to streamline frontend setup in CI workflows.

SOURCE: https://github.com/gradio-app/gradio/blob/main/testing-guidelines/ci.md#_snippet_2

LANGUAGE: APIDOC
CODE:
```
GitHub Action: install-frontend-deps
Purpose: Installs node, pnpm, and builds the frontend.
Source: .github/actions/install-frontend-deps/action.yml
Inputs: (See action.yml for details)
```

----------------------------------------

TITLE: Cancel iterative Gradio jobs mid-stream
DESCRIPTION: Demonstrates how to cancel a job that has iterative outputs, causing it to finish immediately. This example uses a `setTimeout` to simulate cancelling after a delay.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_21

LANGUAGE: JavaScript
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.connect("gradio/count_generator");
const job = app.submit(0, [9]);

for await (const message of job) {
	console.log(message.data);
}

setTimeout(() => {
	job.cancel();
}, 3000);
```

----------------------------------------

TITLE: Install Gradio from main branch
DESCRIPTION: Clones the Gradio repository and runs the installation script for setting up the development environment.

SOURCE: https://github.com/gradio-app/gradio/blob/main/CONTRIBUTING.md#_snippet_0

LANGUAGE: Bash
CODE:
```
bash scripts/install_gradio.sh
```

LANGUAGE: Batch
CODE:
```
scripts\install_gradio.bat
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, the Python package installer. The `-q` flag ensures a quiet installation, suppressing verbose output during the process.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_datetime/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Make Gradio Prediction with Optional Keyword Arguments (Python)
DESCRIPTION: This Python example shows how to interact with Gradio applications where some parameters have default values. It demonstrates calling `Client.predict()` with only required arguments, relying on the app's defaults, and then overriding an optional argument like 'steps'.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_11

LANGUAGE: python
CODE:
```
from gradio_client import Client

client = Client("abidlabs/image_generator")
client.predict(text="an astronaut riding a camel")
```

LANGUAGE: python
CODE:
```
from gradio_client import Client

client = Client("abidlabs/image_generator")
client.predict(text="an astronaut riding a camel", steps=25)
```

----------------------------------------

TITLE: Transcribe Audio with Gradio Client
DESCRIPTION: This snippet demonstrates how to use the `gradio_client` to interact with a hosted Gradio application (e.g., a Whisper model on Hugging Face Spaces). It shows how to instantiate a client and make a prediction by passing an audio file.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_0

LANGUAGE: python
CODE:
```
from gradio_client import Client, handle_file

client = Client("abidlabs/whisper")

client.predict(
    audio=handle_file("audio_sample.wav")
)
```

----------------------------------------

TITLE: Download Example Assets for Gradio Depth Estimation Demo
DESCRIPTION: This Python snippet uses `os` module and `wget` commands to create an 'examples' directory and download an example image and a `packages.txt` file from the Gradio GitHub repository. These assets are used as inputs for the depth estimation demo.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/depth_estimation/run.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
import os
os.mkdir('examples')
!wget -q -O examples/1-jonathan-borba-CgWTqYxHEkg-unsplash.jpg https://github.com/gradio-app/gradio/raw/main/demo/depth_estimation/examples/1-jonathan-borba-CgWTqYxHEkg-unsplash.jpg
!wget -q https://github.com/gradio-app/gradio/raw/main/demo/depth_estimation/packages.txt
```

----------------------------------------

TITLE: Make a prediction with file input using Gradio Client
DESCRIPTION: Illustrates how to handle file inputs (like audio) for predictions by fetching a blob and using `handle_file()` with the Gradio Client. Notes that `Buffer`, `Blob`, or `File` can be used depending on the environment.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_14

LANGUAGE: JavaScript
CODE:
```
import { Client, handle_file } from "@gradio/client";

const response = await fetch(
	"https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3"
);
const audio_file = await response.blob();

const app = await Client.connect("abidlabs/whisper");
const result = await app.predict("/predict", [handle_file(audio_file)]);
```

----------------------------------------

TITLE: Connect to a Public Hugging Face Gradio Space
DESCRIPTION: This snippet shows how to initialize a `Client` object to connect to a publicly available Gradio application hosted on Hugging Face Spaces. Simply provide the Space name to the `Client` constructor.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_2

LANGUAGE: python
CODE:
```
from gradio_client import Client

client = Client("abidlabs/en2fr")  # a Space that translates from English to French
```

----------------------------------------

TITLE: Control Gradio Examples Cache Reset on App Start
DESCRIPTION: If set to 'True', this variable instructs Gradio to delete and recreate the examples cache directory when the app starts. This ensures that cached examples are not reused from previous runs. The default behavior is 'False', meaning existing caches are reused.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/04_additional-features/10_environment-variables.md#_snippet_18

LANGUAGE: sh
CODE:
```
export GRADIO_RESET_EXAMPLES_CACHE="True"
```

----------------------------------------

TITLE: Download Gradio Demo Initialization File
DESCRIPTION: This snippet uses the `wget` command to quietly download the `__init__.py` file from the `clear_components` demo directory within the Gradio GitHub repository. This file is likely a dependency or part of a larger Gradio demo setup.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/clear_components/run.ipynb#_snippet_1

LANGUAGE: bash
CODE:
```
!wget -q https://github.com/gradio-app/gradio/raw/main/demo/clear_components/__init__.py
```

----------------------------------------

TITLE: Install Gradio Python Library
DESCRIPTION: Installs the Gradio library, a Python framework for building machine learning web UIs, using pip. The -q flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/image_editor_sketchpad/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Connect to Authenticated Gradio App
DESCRIPTION: This snippet demonstrates how to connect to a Gradio application that requires authentication. It shows how to pass a username and password as a list to the `auth` parameter of the `Client` constructor.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_6

LANGUAGE: python
CODE:
```
from gradio_client import Client

Client(
  space_name,
  auth=[username, password]
)
```

----------------------------------------

TITLE: Gradio Dataset Basic Initialization
DESCRIPTION: Demonstrates the basic instantiation of the `gradio.Dataset` component.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/gradio/03_components/dataset.svx#_snippet_0

LANGUAGE: python
CODE:
```
gradio.Dataset(···)
```

----------------------------------------

TITLE: Create and Launch Gradio Bar Plot Demo
DESCRIPTION: Initializes a Gradio application that displays a bar plot. It imports the `gradio` library and a DataFrame `df` from `data.py`. A `gr.BarPlot` component is configured to visualize 'weight' on the x-axis (binned by 10) and 'height' on the y-axis (aggregated by sum). The `demo.launch()` method starts the Gradio web interface.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/plot_guide_aggregate_quantitative/run.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
import gradio as gr
from data import df

with gr.Blocks() as demo:
    gr.BarPlot(df, x="weight", y="height", x_bin=10, y_aggregate="sum")

if __name__ == "__main__":
    demo.launch()
```

----------------------------------------

TITLE: Initialize and Launch Gradio GPT-2 XL Demo
DESCRIPTION: This Python code sets up a Gradio interface for the `huggingface/gpt2-xl` model. It defines a title, configures a multi-line text input, and provides example prompts. The `gr.load` function integrates the Hugging Face model, and `demo.launch()` starts the web server for the interactive demo.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/gpt2_xl/run.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
import gradio as gr

title = "gpt2-xl"

examples = [
    ["The tower is 324 metres (1,063 ft) tall,"],
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

demo = gr.load(
    "huggingface/gpt2-xl",
    inputs=gr.Textbox(lines=5, max_lines=6, label="Input Text"),
    title=title,
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()
```

----------------------------------------

TITLE: Configure Gradio Client to receive status events
DESCRIPTION: Shows how to instantiate the Gradio Client with the `events` option, including 'status' and 'data', to ensure status messages are reported back to the client.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_16

LANGUAGE: TypeScript
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.connect("abidlabs/en2fr", {
	events: ["status", "data"]
});
```

----------------------------------------

TITLE: Start Gradio-Lite Development Server
DESCRIPTION: Initiates the development server for Gradio-Lite, allowing local testing of the Pyodide-based library directly in the browser.

SOURCE: https://github.com/gradio-app/gradio/blob/main/CONTRIBUTING.md#_snippet_21

LANGUAGE: bash
CODE:
```
bash scripts/run_lite.sh
```

LANGUAGE: bash
CODE:
```
scripts\run_lite.bat
```

----------------------------------------

TITLE: Prepare Demo Files for Gradio Application
DESCRIPTION: This snippet initializes the local environment by creating a 'files' directory and downloading various media assets (audio, images, video, CSVs) from the Gradio GitHub repository. These files serve as default inputs or examples for the subsequent Gradio interface demonstration.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/kitchen_sink/run.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
import os
os.mkdir('files')
```

LANGUAGE: Bash
CODE:
```
wget -q -O files/cantina.wav https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/cantina.wav
wget -q -O files/cheetah1.jpg https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/cheetah1.jpg
wget -q -O files/lion.jpg https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/lion.jpg
wget -q -O files/logo.png https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/logo.png
wget -q -O files/time.csv https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/time.csv
wget -q -O files/titanic.csv https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/titanic.csv
wget -q -O files/tower.jpg https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/tower.jpg
wget -q -O files/world.mp4 https://github.com/gradio-app/gradio/raw/main/demo/kitchen_sink/files/world.mp4
```

----------------------------------------

TITLE: Full Gradio Blocks Application Example
DESCRIPTION: A complete example showcasing how to build an interactive Gradio application using `gr.Blocks`. It includes defining input and output components, creating a button, handling user interactions with `btn.click()`, and launching the application.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/gradio/01_building-demos/04_blocks.svx#_snippet_1

LANGUAGE: python
CODE:
```
import gradio as gr
def update(name):
    return f"Welcome to Gradio, {name}!"

with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)

demo.launch()
```

----------------------------------------

TITLE: Connect to a General Gradio App URL with JavaScript
DESCRIPTION: This snippet illustrates how to connect the Gradio Client to any Gradio application running at a specific URL. It allows interaction with Gradio apps hosted outside of Hugging Face Spaces, provided the full URL is given.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_8

LANGUAGE: javascript
CODE:
```
import { Client } from "@gradio/client";

const app = Client.connect("https://bec81a83-5b5c-471e.gradio.live");
```

----------------------------------------

TITLE: Make Gradio Prediction with Multiple Positional Arguments (Python)
DESCRIPTION: This Python example illustrates how to pass multiple positional arguments to the `Client.predict()` method when interacting with a Gradio application that expects several inputs. It demonstrates a calculator app scenario where numbers and an operation are provided sequentially.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_9

LANGUAGE: python
CODE:
```
from gradio_client import Client

client = Client("gradio/calculator")
client.predict(4, "add", 5)
```

----------------------------------------

TITLE: Install and Get Help for Gradio CLI Tool
DESCRIPTION: This snippet shows how to install the "gradio" command-line tool using "cargo", Rust's package manager. After installation, it demonstrates how to display the CLI's help message ("gr --help") to understand available commands and options for interacting with Gradio spaces.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/third-party-clients/third-party-clients/rust-client.svx#_snippet_1

LANGUAGE: bash
CODE:
```
cargo install gradio
gr --help
```

----------------------------------------

TITLE: Basic Usage of Gradio Audio Component
DESCRIPTION: Demonstrates the simplest way to initialize the `gradio.Audio` component.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/gradio/03_components/audio.svx#_snippet_0

LANGUAGE: python
CODE:
```
gradio.Audio(···)
```

----------------------------------------

TITLE: Install gradio_client via pip
DESCRIPTION: Instructions for installing the lightweight `gradio_client` library using pip. This command installs the client without the full `gradio` package.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/python-client/gradio_client/01_introduction.svx#_snippet_0

LANGUAGE: Bash
CODE:
```
pip install gradio_client
```

----------------------------------------

TITLE: Install Supabase Python Client
DESCRIPTION: Command-line instruction to install the necessary Supabase Python library using pip, enabling interaction with Supabase databases.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/10_other-tutorials/creating-a-dashboard-from-supabase-data.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install supabase
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, the Python package installer. The `-q` flag ensures a quiet installation, suppressing most output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/multimodaltextbox_component/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Listen for real-time values from Gradio generator endpoints
DESCRIPTION: Shows how to use the iterable interface (`for await...of`) to receive a series of values in real-time from Gradio API endpoints that return multiple values over time.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_20

LANGUAGE: JavaScript
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.connect("gradio/count_generator");
const job = app.submit(0, [9]);

for await (const message of job) {
	console.log(message.data);
}
```

----------------------------------------

TITLE: Make Gradio Prediction with Keyword Arguments (Python)
DESCRIPTION: This Python snippet demonstrates the recommended approach for making predictions using `Client.predict()` by providing arguments as keyword arguments. This method improves readability and allows leveraging default argument values defined in the Gradio application.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_10

LANGUAGE: python
CODE:
```
from gradio_client import Client

client = Client("gradio/calculator")
client.predict(num1=4, operation="add", num2=5)
```

----------------------------------------

TITLE: GitHub Action: install-all-deps
DESCRIPTION: This comprehensive GitHub Action handles the installation of all project dependencies. It first calls the `install-frontend-deps` action, then installs Python, all required project dependencies, and finally installs the Gradio packages locally in editable mode. It also includes logic to manage discrepancies between Windows and Linux environments.

SOURCE: https://github.com/gradio-app/gradio/blob/main/testing-guidelines/ci.md#_snippet_3

LANGUAGE: APIDOC
CODE:
```
GitHub Action: install-all-deps
Purpose: Calls install-frontend-deps, then installs python, all dependencies, and gradio packages locally (editable mode). Handles Windows/Linux environment discrepancies.
Source: .github/actions/install-all-deps/action.yml
Inputs: (See action.yml for details)
```

----------------------------------------

TITLE: Install Gradio Python Library
DESCRIPTION: Installs the Gradio library, a Python framework for building machine learning web UIs, using pip. The `-q` flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/chatbot_multimodal/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip. The `-q` flag ensures a quiet installation, suppressing most output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/render_heavy_concurrently/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Make a prediction with multiple parameters using Gradio Client
DESCRIPTION: Shows how to pass multiple parameters as an array to the `.predict()` method when interacting with a Gradio application.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_13

LANGUAGE: JavaScript
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.connect("gradio/calculator");
const result = await app.predict("/predict", [4, "add", 5]);
```

----------------------------------------

TITLE: Download sample audio files for Gradio demo
DESCRIPTION: Creates an 'audio' directory if it doesn't exist and downloads two sample WAV audio files ('cantina.wav' and 'recording1.wav') from the Gradio GitHub repository. These files serve as pre-loaded examples for the Gradio interface.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/main_note/run.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
import os
os.mkdir('audio')
!wget -q -O audio/cantina.wav https://github.com/gradio-app/gradio/raw/main/demo/main_note/audio/cantina.wav
!wget -q -O audio/recording1.wav https://github.com/gradio-app/gradio/raw/main/demo/main_note/audio/recording1.wav
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a package installer for Python. The `-q` flag ensures a quiet installation, suppressing most output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/audio_debugger/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a package installer for Python. The -q flag ensures a quiet installation, suppressing most output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/input_output/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Prepare Gradio Demo Environment and Download Media Files
DESCRIPTION: This Python snippet sets up the environment for the Gradio demo by creating a 'files' directory. It then uses shell commands (executed via `!wget`) to download sample image, audio, and video files from the Gradio GitHub repository, which are subsequently used by the Gradio application's media components.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/change_vs_input/run.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
import os
os.mkdir('files')
!wget -q -O files/cantina.wav https://github.com/gradio-app/gradio/raw/main/demo/change_vs_input/files/cantina.wav
!wget -q -O files/lion.jpg https://github.com/gradio-app/gradio/raw/main/demo/change_vs_input/files/lion.jpg
!wget -q -O files/world.mp4 https://github.com/gradio-app/gradio/raw/main/demo/change_vs_input/files/world.mp4
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip. The `-q` flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/model3d_component/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: This snippet installs the Gradio library using pip, the Python package installer. The '-q' flag ensures a quiet installation, suppressing verbose output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/hello_world_4/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Gradio Interface `log.csv` Example File
DESCRIPTION: This CSV file demonstrates the required structure for `log.csv` when loading examples from a directory for a Gradio `Interface` with multiple inputs. Each row represents a distinct example, and the column headers correspond to the input parameters of the prediction function, allowing the interface to map example values correctly.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/cn/02_building-interfaces/03_more-on-examples.md#_snippet_0

LANGUAGE: csv
CODE:
```
num,operation,num2
5,"add",3
4,"divide",2
5,"multiply",3
```

----------------------------------------

TITLE: Check cURL Installation Version
DESCRIPTION: Executes the curl --version command to verify if cURL is installed on the system and to display its current version details. This is a common first step for troubleshooting cURL setup.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/03_querying-gradio-apps-with-curl.md#_snippet_3

LANGUAGE: bash
CODE:
```
curl --version
```

----------------------------------------

TITLE: Install Python dependencies for Gradio XGBoost demo
DESCRIPTION: Installs necessary Python libraries including gradio, numpy, matplotlib, shap, xgboost, pandas, and datasets. This setup is crucial for running the income prediction and explainability demo.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/xgboost-income-prediction-with-explainability/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio numpy==1.23.2 matplotlib shap xgboost==1.7.6 pandas datasets
```

----------------------------------------

TITLE: Install Gradio Python Library
DESCRIPTION: Installs the Gradio library using pip, the Python package installer. The -q flag suppresses output during installation, making it quiet.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/chatbot_simple/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Iterate Over Gradio Generator Job Outputs
DESCRIPTION: Demonstrates how to directly iterate over a `Job` object to process outputs from a Gradio generator endpoint as they are returned. This provides a streaming-like experience for real-time output display, processing each value as it becomes available.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_18

LANGUAGE: Python
CODE:
```
from gradio_client import Client

client = Client(src="gradio/count_generator")
job = client.submit(3, api_name="/count")

for o in job:
    print(o)
```

----------------------------------------

TITLE: Example Public Google Sheet URL
DESCRIPTION: An example URL for a public Google Sheet, demonstrating the format before modification for CSV export. This URL is obtained by using the 'Get shareable link' option in Google Sheets.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/10_other-tutorials/creating-a-realtime-dashboard-from-google-sheets.md#_snippet_0

LANGUAGE: html
CODE:
```
https://docs.google.com/spreadsheets/d/1UoKzzRzOCt-FXLLqDKLbryEKEgllGAQUEJ5qtmmQwpU/edit#gid=0
```

----------------------------------------

TITLE: Launch Gradio App as Progressive Web App (PWA)
DESCRIPTION: This Python example demonstrates how to launch a Gradio application as a Progressive Web App (PWA) by setting the `pwa=True` parameter in the `launch()` method. This allows the web app to be installed like a native application.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/04_additional-features/07_sharing-your-app.md#_snippet_24

LANGUAGE: Python
CODE:
```
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

demo.launch(pwa=True)  # Launch your app as a PWA
```

----------------------------------------

TITLE: Transcribe Audio with Gradio JavaScript Client
DESCRIPTION: Demonstrates how to use the @gradio/client library to connect to a Gradio app hosted on Hugging Face Spaces and transcribe an audio file programmatically. It fetches an audio file, connects to the 'abidlabs/whisper' Space, and predicts the transcription.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_0

LANGUAGE: js
CODE:
```
import { Client, handle_file } from "@gradio/client";

const response = await fetch(
	"https://github.com/audio-samples/audio-samples.github.io/raw/master/samples/wav/ted_speakers/SalmanKhan/sample-1.wav"
);
const audio_file = await response.blob();

const app = await Client.connect("abidlabs/whisper");
const transcription = await app.predict("/predict", [handle_file(audio_file)]);

console.log(transcription.data);
// [ "I said the same phrase 30 times." ]
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: This snippet installs the Gradio library using pip, the standard package installer for Python. The `-q` flag ensures a quiet installation, suppressing verbose output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/hello_blocks_decorator/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip. The '-q' flag ensures a quiet installation, suppressing most output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/textbox_component/run.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Cancel Gradio jobs using the job instance
DESCRIPTION: Illustrates how to use the `.cancel()` method on a job instance to cancel queued or running Gradio jobs. Explains the behavior for jobs that have started versus those still in the queue.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/02_getting-started-with-the-js-client.md#_snippet_19

LANGUAGE: JavaScript
CODE:
```
import { Client } from "@gradio/client";

const app = await Client.connect("abidlabs/en2fr");
const job_one = app.submit("/predict", ["Hello"]);
const job_two = app.submit("/predict", ["Friends"]);

job_one.cancel();
job_two.cancel();
```

----------------------------------------

TITLE: Make Gradio Prediction with File or URL Inputs (Python)
DESCRIPTION: This Python snippet demonstrates how to provide file paths or URLs as inputs to a Gradio application using `Client.predict()`. It highlights the use of `gradio_client.handle_file()` to correctly upload and preprocess the input file for the Gradio server.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_12

LANGUAGE: python
CODE:
```
from gradio_client import Client, handle_file

client = Client("abidlabs/whisper")
client.predict(
    audio=handle_file("https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3")
)
```

----------------------------------------

TITLE: gradio.DownloadData Basic Instantiation
DESCRIPTION: Shows the fundamental syntax for initializing the `gradio.DownloadData` component, typically used as a type hint or placeholder in function signatures.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/gradio/04_helpers/10_downloaddata.svx#_snippet_0

LANGUAGE: python
CODE:
```
gradio.DownloadData(···)
```

----------------------------------------

TITLE: Gradio LinePlot Embedded Full Python Example
DESCRIPTION: A comprehensive example demonstrating the gradio.LinePlot component within a gradio-lite block. It shows how to prepare data using pandas and numpy, and then configure gr.LinePlot with specific x, y, title, and container properties. The example includes the necessary Python imports and component setup within a gr.Blocks context.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/gradio/03_components/lineplot.svx#_snippet_2

LANGUAGE: python
CODE:
```
import gradio as gr
import pandas as pd
import numpy as np
simple = pd.DataFrame(np.array(
    [
        [1, 28],
        [2, 55],
        [3, 43],
        [4, 91],
        [5, 81],
        [6, 53],
        [7, 19],
        [8, 87],
        [9, 52]
    ]
), columns=["week", "price"])
with gr.Blocks() as demo:
    gr.LinePlot(
        value=simple,
        x="week",
        y="price",
        title="Stock Price Chart",
        container=True,
        width=400
    )
demo.launch()
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip. The `-q` flag ensures a quiet installation, suppressing verbose output, which is useful for cleaner script execution.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/highlightedtext_component/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a package installer for Python. The `-q` flag ensures a quiet installation, suppressing verbose output.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/gallery_component/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a common first step for setting up Gradio projects. The '-q' flag ensures a quiet installation.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/latex/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Initialize Gradio Custom Component Project
DESCRIPTION: Use the `gradio cc create` command to scaffold a new custom component project. This command creates a subdirectory with the specified name (e.g., 'pdf') containing the necessary frontend, backend, and demo files.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/08_custom-components/07_pdf-component-example.md#_snippet_0

LANGUAGE: bash
CODE:
```
gradio cc create PDF
```

----------------------------------------

TITLE: Create a new Gradio custom component template
DESCRIPTION: Initializes a new custom component project with a specified name and template. The `SimpleTextbox` template is recommended for beginners, providing a stripped-down version of the `Textbox` component to start development.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/08_custom-components/01_custom-components-in-five-minutes.md#_snippet_0

LANGUAGE: bash
CODE:
```
gradio cc create MyComponent --template SimpleTextbox
```

----------------------------------------

TITLE: Cancel Gradio Client Jobs
DESCRIPTION: Illustrates how to cancel a queued Gradio client job using the `.cancel()` method. This method returns `True` if the job was successfully canceled (i.e., not yet started) and `False` otherwise, indicating it has already begun processing.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/01_getting-started-with-the-python-client.md#_snippet_16

LANGUAGE: Python
CODE:
```
client = Client("abidlabs/whisper")
job1 = client.submit(handle_file("audio_sample1.wav"))
job2 = client.submit(handle_file("audio_sample2.wav"))
job1.cancel()  # will return False, assuming the job has started
job2.cancel()  # will return True, indicating that the job has been canceled
```

----------------------------------------

TITLE: Install Gradio Library
DESCRIPTION: Installs the Gradio library using pip, a Python package installer. This is a common first step for setting up and running Gradio applications in an environment.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/timer_simple/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Gradio Prebuilt Themes API Reference
DESCRIPTION: Reference for prebuilt themes available in Gradio's `gr.themes` module, including their names and brief descriptions of their visual characteristics and primary use cases.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/10_other-tutorials/theming-guide.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
gr.themes.Base(): The "base" theme, minimal styling, useful as a base for custom themes.
gr.themes.Default(): The "default" Gradio 5 theme, vibrant orange primary, gray secondary.
gr.themes.Origin(): The "origin" theme, similar to Gradio 4 styling, subdued colors.
gr.themes.Citrus(): The "citrus" theme, yellow primary, highlights focused elements, 3D button effects.
gr.themes.Monochrome(): The "monochrome" theme, black primary, white secondary, serif fonts.
gr.themes.Soft(): The "soft" theme, purple primary, white secondary, increased border radius, highlights labels.
gr.themes.Glass(): The "glass" theme, blue primary, translucent gray secondary, vertical gradients for glassy effect.
gr.themes.Ocean(): The "ocean" theme, blue-green primary, gray secondary, horizontal gradients for buttons and form elements.
```

----------------------------------------

TITLE: Customize Gradio ChatInterface with various arguments
DESCRIPTION: This Python example demonstrates how to create a `gr.ChatInterface` with a custom backend function (`yes_man`) and apply various customizations. It sets the chatbot height, textbox placeholder, title, description, theme, and includes pre-defined examples with caching enabled. This showcases a comprehensive setup for a Gradio chatbot.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/05_chatbots/01_creating-a-chatbot-fast.md#_snippet_11

LANGUAGE: Python
CODE:
```
import gradio as gr

def yes_man(message, history):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"

gr.ChatInterface(
    yes_man,
    type="messages",
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="Yes Man",
    description="Ask Yes Man any question",
    theme="ocean",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
).launch()
```

----------------------------------------

TITLE: Install Gradio Library via Pip
DESCRIPTION: Installs the Gradio library using pip, Python's package installer. The -q flag ensures a quiet installation, suppressing verbose output. This is a prerequisite for running Gradio applications.

SOURCE: https://github.com/gradio-app/gradio/blob/main/demo/image_editor_webcam/run.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install -q gradio
```

----------------------------------------

TITLE: Launch Gradio Theme Builder locally
DESCRIPTION: Shows how to import the Gradio library and launch the interactive Theme Builder tool locally, which allows real-time preview and generation of custom theme code.

SOURCE: https://github.com/gradio-app/gradio/blob/main/guides/10_other-tutorials/theming-guide.md#_snippet_2

LANGUAGE: Python
CODE:
```
import gradio as gr

gr.themes.builder()
```

----------------------------------------

TITLE: Basic Instantiation of `gradio.Sidebar`
DESCRIPTION: Shows the fundamental way to create an instance of the `gradio.Sidebar` component.

SOURCE: https://github.com/gradio-app/gradio/blob/main/js/_website/src/lib/templates/gradio/03_components/sidebar.svx#_snippet_0

LANGUAGE: python
CODE:
```
gradio.Sidebar(···)
```
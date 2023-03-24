# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import requests
from collections import OrderedDict
from modules import scripts, sd_models, shared
import json
import os.path
import shutil

sapi_dir = scripts.basedir()

CLIENT_AGENT = "Stability API for Web UI:beta:TwoDukes & TwoPerCent"
settings_file = os.path.join(scripts.basedir(), "stability_settings.json")

shutil.copy2(os.path.join(sapi_dir, 'Dummy.safetensors'), os.path.join(sd_models.model_path, 'Set_Up_Your_API_Key.SAPI.safetensors'))

class API_Manager():

    session = requests.Session()
    apis = OrderedDict()
    api_key = None
    api_endpoint = None
    credits = 0.0
    remove_non_api_components = False
    email = ''
    
    NON_API_COMPONENTS = [
        "txt2img_enable_hr", 
        "img2img_restore_faces", 
        "txt2img_restore_faces", 
        "img2img_tiling", 
        "txt2img_tiling",
        "resize_mode",
        "img2img_inpainting_fill",
        "img2img_inpaint_full_res"
    ]

    def __init__(self):
        self.load_settings()
        self.update_current_api(list(self.apis.keys())[0])

    def toggle_remove_components(self):
        self.remove_non_api_components = not self.remove_non_api_components
        self.save_settings()

    def reset_settings(self):
        self.apis.clear()
        self.apis["https://api.stability.ai/v1/"] = "0000000000000"
        self.save_settings()
        self.load_settings()

    def load_settings(self):
        if os.path.exists(settings_file):
            with open(settings_file) as file:
                settings = json.load(file)
                self.apis.clear()
                for endpoint, key in settings["endpoint_keys"].items():
                    self.apis[endpoint] = key
                self.remove_non_api_components = settings["settings"]["hide_components"]
        else:
            self.reset_settings()
        return self.apis

    def update_current_api(self, endpoint):
        self.credits = 0.0
        self.api_endpoint = endpoint
        self.api_key = self.apis[endpoint]
        self.apis.move_to_end(endpoint, last=False)
        self.load_models()
        return self.api_key

    def update_key(self, api, new_key):
        self.apis[api] = new_key 
        self.api_key = new_key
        self.save_settings()
        self.load_models()

    def load_models(self):
        try:
            for filename in os.listdir(sd_models.model_path):
                if "SAPI.safetensors" in filename:
                    os.remove(os.path.join(sd_models.model_path, filename))  
            models = requests.get("{}engines/list".format(self.api_endpoint), headers={"Authorization": self.api_key})  
            models = models.json()
            formatted_models = ["{}".format(m["id"]) for m in models]
            for model in formatted_models:
                shutil.copy2(os.path.join(sapi_dir, 'Dummy.safetensors'), os.path.join(sd_models.model_path, model + '.SAPI.safetensors'))
            shared.refresh_checkpoints()
            sd_models.list_models()
        except Exception as e:
            models = []
            formatted_models = []
        finally:
            for filename in os.listdir(sd_models.model_path):
                if filename.endswith("SAPI.safetensors"):
                    break
            else:
                shutil.copy2(os.path.join(sapi_dir, 'Dummy.safetensors'), os.path.join(sd_models.model_path, 'Set_up_your_API_Key.SAPI.safetensors')) # Ensures users don't ever get stuck with no models
                shared.refresh_checkpoints()
                sd_models.list_models()
   
    def save_settings(self):
        opts = {
            "endpoint_keys": {},
            "settings": {
                        "hide_components": self.remove_non_api_components,
            },

        }
        for k,v in self.apis.items():
            if v is not None:
                opts["endpoint_keys"][k] = v
        with open(settings_file, "w") as file:
            json.dump(opts, file)

api_manager = API_Manager()


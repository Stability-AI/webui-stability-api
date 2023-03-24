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

from modules import script_callbacks
import gradio
from modules import scripts, shared
from scripts.settings_manager import api_manager as manager

sapi_dir = scripts.basedir()

class Settings():

    def ui(self):
        manager.load_settings()

        def update_click():
            response = manager.session.get("{}user/balance".format(manager.api_endpoint), headers={"Authorization": manager.api_key})
            assert response.status_code == 200, "Status Code: {} (expected {})".format(response.status_code, 200)
            manager.credits = "{:.2f}".format(response.json()["credits"])
            print("Credits: {}".format(manager.credits))

            response = manager.session.get("{}user/account".format(manager.api_endpoint), headers={"Authorization": manager.api_key})
            assert response.status_code == 200, "Status Code: {} (expected {})".format(response.status_code, 200)
            manager.email = response.json()["email"]

            return "Account Credits: {}".format(manager.credits), "Account Email: {}".format(manager.email)
        
        with gradio.Blocks(analytics_enabled=False) as settings_tab:   

            with gradio.Row():
                api_list = gradio.Dropdown(choices=[endpoint for endpoint, api in manager.apis.items()], value=list(manager.apis.keys())[0], label="API Endpoint", interactive=True)   
                
            with gradio.Row():
                api_key_text = gradio.Textbox(manager.api_key, max_lines=1, placeholder="0000000000", label="API key", interactive=True, type="password")
                api_key_text.change(fn=manager.update_key, inputs=[api_list, api_key_text], outputs=None)
                
                show = gradio.Button(value="Show API Key", elem_id="stability_show_api_key")
                def show_click(show):
                    return (gradio.update(type="text" if show == "Show API Key" else "password"), gradio.update(value="Hide" if show == "Show API Key" else "Show API Key"))
                show.click(fn=show_click, inputs=show, outputs=[api_key_text, show])

                api_list.change(fn=manager.update_current_api, inputs=api_list, outputs=api_key_text)

                manager.api_endpoint = api_list.value
                manager.api_key = manager.apis[api_list.value]
                api_key_text.value = manager.api_key

            with gradio.Row(elem_id="stability_account_row"):
                account = gradio.Textbox("Account Email: {}".format(manager.email), show_label=False,  elem_id="stability_account", interactive=False)
                account.style(container=False)
                credits = gradio.Textbox("Account Credits: {}".format(manager.credits), show_label=False, elem_id="stability_credits", interactive=False)
                credits.style(container=False)
                update = gradio.Button("\U0001f504", elem_id="stability_update")  
                update.click(fn=update_click, outputs=[credits, account])

                api_key_text.change(fn=update_click, outputs=[credits, account])
            with gradio.Column():
                gradio.HTML("1. Get your key at <a href='https://beta.dreamstudio.ai/membership?tab=apiKeys' target='_blank'>https://beta.dreamstudio.ai/membership?tab=apiKeys</a>")
                gradio.HTML("2. Refresh checkpoint list after authentication to get available checkpoints")
                gradio.HTML("3. Generate without making your GPU go brrrr!")
   
            with gradio.Row():
                remove_non_api_checkbox = gradio.Checkbox(interactive=True, value=manager.remove_non_api_components, label="Hide Non-API Components (Restarts UI)", elem_id="stability_remove_non_api_components")
                remove_non_api_checkbox.change(fn=manager.toggle_remove_components)

            def request_restart():
                shared.state.interrupt()
                shared.state.need_restart = True

            remove_non_api_checkbox.change(
                fn=request_restart,
                _js='restart_reload',
                inputs=[],
                outputs=[],
            )              

            try:
                update_click()
            except:
                pass

        return [(settings_tab, "Stability API Settings", "stability_api_settings")]
    
settings = Settings()
script_callbacks.on_ui_tabs(settings.ui)
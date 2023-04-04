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

import math
from PIL import Image
from modules import scripts, processing, shared, images, devices, lowvram
from modules.shared import opts, state
import modules
import gradio
import requests
import PIL.Image
from PIL import ImageOps
import base64
import io
import os.path
import numpy
import itertools
import torch   
from scripts.settings_manager import api_manager as manager

Image.MAX_IMAGE_PIXELS = None


class FakeCheckpointInfo:
    def __init__(self, model_name):
        self.model_name = model_name

class FakeModel:
    sd_model_hash=""

    def __init__(self, name):
        self.sd_checkpoint_info = FakeCheckpointInfo(name)

class StabilityGenerateError(Exception):
    pass

class SAPIExtensionGenerateError(Exception):
    pass

cg = False
cg_i2i = False
error_message = None
info_message = None

class Main(scripts.Script):
    TITLE = "Run on Stability API"
    SAMPLERS = {
        "DDIM": "DDIM",
        "DDPM": "DDPM",
        "DPM++ 2M Karras": "K_DPMPP_2M",
        "DPM++ 2S a Karras": "K_DPMPP_2S_ANCESTRAL",
        "DPM2 Karras": "K_DPM_2",
        "DPM2 a Karras": "K_DPM_2_ANCESTRAL",
        "Euler": "K_EULER",
        "Euler a": "K_EULER_ANCESTRAL",
        "Heun": "K_HEUN",
        "LMS Karras": "K_LMS"
    }

    proccessed = None
    processing = False
    use_api = False

    def title(self):
        return self.TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def update_click(self):
        try:
            response = manager.session.get("{}user/balance".format(manager.api_endpoint), headers={"Authorization": manager.api_key})
            assert response.status_code == 200, "Status Code: {} (expected {})".format(response.status_code, 200)
            prev_credits = manager.credits
            manager.credits = "{:.2f}".format(response.json()["credits"])
            response = manager.session.get("{}user/account".format(manager.api_endpoint), headers={"Authorization": manager.api_key})
            assert response.status_code == 200, "Status Code: {} (expected {})".format(response.status_code, 200)
            manager.email = response.json()["email"]
            used = ''
            if float(prev_credits) != 0 and float(prev_credits) != float(manager.credits):
                used = ', Last Request Cost: {}'.format(round(float(prev_credits) - float(manager.credits),2))
            return "Stability API Account Credits: {}".format(manager.credits) + used
        except Exception as e:
            return "Don't forget to set up your Stability API key!"

    def halt_process(self, p, message):
        global error_message
        #print(f"Processing error {message}")
        p.n_iter = 0
        p.disable_extra_networks = True
        state.interrupted = True
        self.processing = False
        error_message = message
        raise StabilityGenerateError(message)
   

    def process(self, p):
        global cg, cg_i2i
        model_name = os.path.basename(shared.sd_model.sd_checkpoint_info.filename).lower().split('.')[0]
        is_api_model = "sapi" in os.path.basename(shared.sd_model.sd_checkpoint_info.filename).lower().split('.')[1]
        
        if not is_api_model:
            self.use_api = False
            return p
        
        if self.processing is False:
            self.use_api = True
            all_prompts_before = p.all_prompts 
            self.proccessed = self.process_images(p, model_name)
            p.n_iter = 0
            p.disable_extra_networks = True
            p.all_prompts = all_prompts_before


    def postprocess(self, p, processed, *args):
        global error_message
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """
        if self.processing is False and self.use_api and error_message is None:
            processed.images = self.proccessed.images
            
    def update_cg_i2i(self, new_cg_i2i):
        global cg_i2i
        cg_i2i = new_cg_i2i

    def update_cg(self, new_cg):
        global cg
        cg= new_cg

    def update_api_message(self):
        global error_message, info_message
        if error_message:
            return "\tStability API: {}".format(error_message)
        if info_message:
            return "\tStability API Info: {}".format(info_message)
        return ""

    def after_component(self, component, **kwargs):
        for k, v in kwargs.items():
            if k == "elem_id":
                if v == ("html_info_txt2img"):
                    with gradio.Row(elem_id="stability_api_message_row"):
                        api_errors = gradio.HTML(label="SAPI Messages", show_label=True, elem_id="stability_api_messages", interactive=False)
                        component.change(self.update_api_message, inputs=None, outputs=api_errors)
                    with gradio.Row(elem_id="stability_credits_row"):
                        credits = gradio.Textbox(self.update_click(), show_label=False, elem_id="stability_credits", interactive=False)
                        credits.style(container=False)
                        update = gradio.Button("\U0001f504", elem_id="stability_update")
                        update.click(fn=self.update_click, outputs=credits)
                        component.change(fn=self.update_click, outputs=credits)    

                if v == ("html_info_img2img"):
                    with gradio.Row(elem_id="stability_api_message_row_img2img"):
                        api_errors_img2img = gradio.HTML(label="SAPI Messages", show_label=True, elem_id="stability_api_messages_img2img", interactive=False)
                        component.change(self.update_api_message, inputs=None, outputs=api_errors_img2img)   
                    with gradio.Row(elem_id="stability_credits_row"):
                        credits_img2img = gradio.Textbox(self.update_click(), show_label=False, elem_id="stability_credits", interactive=False)
                        credits_img2img.style(container=False)
                        update_img2img = gradio.Button("\U0001f504", elem_id="stability_update")
                        update_img2img.click(fn=self.update_click, outputs=credits_img2img)
                        component.change(fn=self.update_click, outputs=credits_img2img)         
                if v == "img2img_tiling":
                    global cg_i2i
                    cg_i2i_box = gradio.Checkbox(value=False, label="Clip Guidance (SAPI Only)", elem_id="stability_clip_guidance_img2img", visible=True, interactive=True)
                    cg_i2i_box.change(self.update_cg_i2i, inputs=cg_i2i_box)
                if v == "txt2img_enable_hr":
                    global cg
                    cg_box = gradio.Checkbox(value=False, label="Clip Guidance (SAPI Only)", elem_id="stability_clip_guidance", visible=True, interactive=True)
                    cg_box.change(self.update_cg, inputs=cg_box)
                    
                if v in manager.NON_API_COMPONENTS and manager.remove_non_api_components:
                    component.visible = False
                    component.interactive = False
                    component.disabled = True
    pass

    def process_images(self, p, model):
        global cg, cg_i2i, info_message, error_message
        
        if p.sampler_name not in self.SAMPLERS.keys():
            self.halt_process(p, "Sampler {} not supported by API, please use one of the following: {}".format(p.sampler_name, ", ".join(self.SAMPLERS.keys())))

        
        devices.torch_gc()
        self.processing = True
        error_message = None
        info_message = None
        self.use_api = True

        stored_opts = {k: shared.opts.data[k] for k in p.override_settings.keys()}

        try:
            for k, v in p.override_settings.items():
                setattr(shared.opts, k, v)

            p.extra_generation_params = {
                "Model": model,
                "Clip Guidance": (cg_i2i if type(p) == processing.StableDiffusionProcessingImg2Img else cg),
                "Face restoration": (opts.face_restoration_model if p.restore_faces else None)
            }

            res = self.process_images_inner(p, model)

        finally:
            if p.override_settings_restore_afterwards:
                for k, v in stored_opts.items():
                    setattr(shared.opts, k, v)
        return res

    def process_images_inner(self, p, model):
        # Copyright (C) 2023  AUTOMATIC1111
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/00dab8f10defbbda579a1bc89c8d4e972c58a20d/modules/processing.py#L501-L717
    
        fake_model = FakeModel(model)
        if type(p) == processing.StableDiffusionProcessingImg2Img:
            self.is_img2img = True
        else:
            self.is_img2img = False

        if "inpainting" in model and (self.is_img2img is False or getattr(p, "image_mask", None) is None): 
                self.halt_process(p, "Inpainting model is for inpainting only")
                

        if p.n_iter > 10:
            self.halt_process(p, "Please keep batch count <= 10")
        
        if type(p.prompt) == list:
            assert(len(p.prompt) > 0)
        else:
            assert p.prompt is not None

        devices.torch_gc()
        seed = processing.get_fixed_seed(p.seed)
        p.subseed = -1
        p.subseed_strength = 0
        p.seed_resize_from_h = 0
        p.seed_resize_from_w = 0

        if not self.is_img2img and "768" in model and p.height * p.width < 589824:
                print(f"Image size is only {p.height * p.width} which is less than 589,824 pixels")
                ratio = p.height / p.width
                old_width = p.width
                old_height = p.height
                new_width = ((589824 / ratio) ** 0.5)
                new_height = ratio * new_width
                p.width = math.ceil(new_width)
                p.height = math.ceil(new_height)
                print(f"Upsizing image request to {p.width}x{p.height} in order to meet minimum image size of 589,824 pixels")
                global info_message
                info_message = "Requested image size of {}x{} was too small for {}, image was resized to {}x{}".format(str(old_width), str(old_height), model, str(p.width), str(p.height))

        if type(p.prompt) == list:
            p.all_prompts = list(itertools.chain.from_iterable((p.batch_size * [shared.prompt_styles.apply_styles_to_prompt(p.prompt[x * p.batch_size], p.styles)] for x in range(p.n_iter))))
        else:
            p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

        if type(p.negative_prompt) == list:
            p.all_negative_prompts = list(itertools.chain.from_iterable((p.batch_size * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt[x * p.batch_size], p.styles)] for x in range(p.n_iter))))
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

        if type(seed) == list:
            p.all_seeds = list(itertools.chain.from_iterable(([seed[x * p.batch_size] + y * seed_variation for y in range(p.batch_size)] for x in range(p.n_iter))))
        else:
            p.all_seeds = [int(seed) + x * 1 for x in range(len(p.all_prompts))] 

        p.all_subseeds = [-1 for _ in range(len(p.all_prompts))]

        def infotext(iteration=0, position_in_batch=0):
            global error_message
            if error_message:
                return
            old_model = shared.sd_model
            shared.sd_model = fake_model
            ret = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, {}, iteration, position_in_batch)
            shared.sd_model = old_model
            return ret

        if p.scripts is not None:
            p.scripts.process(p)
        
        infotexts = []
        output_images = []

        with torch.no_grad():
            with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
                old_model = shared.sd_model
                shared.sd_model = fake_model
                processed = processing.Processed(p, [], p.seed, "")
                file.write(processed.infotext(p, 0))
                shared.sd_model = old_model

            if shared.state.job_count == -1:
                shared.state.job_count = p.n_iter

            for n in range(p.n_iter):
                p.iteration = n

                if shared.state.skipped:
                    shared.state.skipped = False

                if shared.state.interrupted:
                    break

                prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
                subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

                if len(prompts) == 0:
                    break

                if p.scripts is not None:
                    p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

                if p.n_iter > 1:
                    shared.state.job = f"Batch {n+1} out of {p.n_iter}"

                x_samples_ddim = self.process_batch_stability(p, model, prompts[0], negative_prompts[0], seeds[0])

                if x_samples_ddim is None or len(x_samples_ddim) == 0:
                    break

                x_samples_ddim = [s.cpu() for s in x_samples_ddim]
                x_samples_ddim = torch.stack(x_samples_ddim).float()

                if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                    lowvram.send_everything_to_cpu()

                devices.torch_gc()

                if p.scripts is not None:
                    p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * numpy.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(numpy.uint8)

                    if p.restore_faces:
                        if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                            images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                        devices.torch_gc()

                        x_sample = modules.face_restoration.restore_faces(x_sample)
                        devices.torch_gc()
                    
                    image = PIL.Image.fromarray(x_sample)

                    if p.scripts is not None:
                        pp = scripts.PostprocessImageArgs(image)
                        p.scripts.postprocess_image(p, pp)
                        image = pp.image
                    

                    if p.color_corrections is not None and i < len(p.color_corrections):
                        if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_color_correction:
                            image_without_cc = processing.apply_overlay(image, p.paste_to, i, p.overlay_images)
                            images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")

                        image = processing.apply_color_correction(p.color_corrections[i], image)

                    image = processing.apply_overlay(image, p.paste_to, i, p.overlay_images)

                    if shared.opts.samples_save and not p.do_not_save_samples:
                        images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=infotext(n, i), p=p)

                    text = infotext(n, i)
                    infotexts.append(text)

                    if shared.opts.enable_pnginfo:
                        image.info["parameters"] = text

                    output_images.append(image)

                del x_samples_ddim
                devices.torch_gc()
                shared.state.job_no += 1
                shared.state.sampling_step = 0
                shared.state.current_image_sampling_step = 0

            p.color_corrections = None
            index_of_first_image = 0
            unwanted_grid_because_of_img_count = len(output_images) < 2 and shared.opts.grid_only_if_multiple

            if (shared.opts.return_grid or shared.opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
                grid = images.image_grid(output_images, p.batch_size)

                if shared.opts.return_grid:
                    text = infotext()
                    infotexts.insert(0, text)

                    if shared.opts.enable_pnginfo:
                        grid.info["parameters"] = text

                    output_images.insert(0, grid)
                    index_of_first_image = 1

                if shared.opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=infotext(), short_filename=not shared.opts.grid_extended_filename, p=p, grid=True)

        devices.torch_gc()
        res = processing.Processed(p, output_images, p.all_seeds[0], infotext(), subseed=-1, index_of_first_image=index_of_first_image, infotexts=["test"])

        if p.scripts is not None:
            p.scripts.postprocess(p, res)

        self.processing = False

        return res

        
    def process_batch_stability(self, p, model, prompt, negative_prompt, seed):

        def round_to_64(num):
            return (num + 63) & ~63

        def resize_image(image):
            max_dim = 1024
            width, height = image.size
            aspect_ratio = width / height
            if width * height > 700000:
                resize_ratio = max_dim / max(width, height)
                new_width = int(round(width * resize_ratio))
                new_height = int(round(height * resize_ratio))
                new_width = int(math.ceil(new_width / 64.0) * 64)  # round up to nearest multiple of 64
                new_height = int(math.ceil(new_height / 64.0) * 64)  # round up to nearest multiple of 64
                width, height = new_width, new_height
            return width, height, image.resize((width, height), PIL.Image.LANCZOS)


        payload = {
                "text_prompts": [
                {"text": prompt, "weight": 1.0}
                ] + (
                [{"text": negative_prompt, "weight": -1.0}] if len(negative_prompt) > 0 else []
                ),
            "cfg_scale": p.cfg_scale,
            "seed": seed,
            "height": round_to_64(p.height),
            "width": round_to_64(p.width),
            "steps": p.steps,
            "samples": p.batch_size,
            "clip_guidance_preset": "FAST_BLUE" if p.extra_generation_params["Clip Guidance"] else "NONE",
            "sampler": self.SAMPLERS.get(p.sampler_name) 
        }

        p.extra_generation_params["Sampler"] = p.sampler_name
        
        files = {}
        
        if self.is_img2img:
            _, _, image = resize_image(p.init_images[0])

            img2img_payload={
                "text_prompts[0][text]": prompt,
                "text_prompts[0][weight]": 1.0,

                "cfg_scale": payload["cfg_scale"],
                "seed": payload["seed"],
                "steps": payload["steps"],
                "samples": payload["samples"],
                "clip_guidance_preset": "FAST_BLUE" if p.extra_generation_params["Clip Guidance"] else "NONE",
                "sampler": payload["sampler"]
            }

            if len(negative_prompt) > 0:
                img2img_payload["text_prompts[1][text]"] = negative_prompt 
                img2img_payload["text_prompts[1][weight]"] = -1.0 
                
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            files["init_image"] = buffer.getvalue()

            if p.image_mask is not None:
                _, _, image = resize_image(p.image_mask)
                if not p.inpainting_mask_invert:
                    image = ImageOps.invert(image)
                buffer2 = io.BytesIO()
                image.save(buffer2, format="PNG")
                files["mask_image"] = buffer2.getvalue()
                img2img_payload["mask_source"] = "MASK_IMAGE_BLACK"
            else:
                img2img_payload["image_strength"] = 1 - p.denoising_strength,
            
            payload = img2img_payload

        if shared.state.skipped or shared.state.interrupted:
            return (None, None)

        try:
            gen_type = "text-to-image" if not self.is_img2img else "image-to-image"
            masking = "/masking" if self.is_img2img and p.image_mask is not None else ''

            url = "{}generation/{}/{}{}".format(manager.api_endpoint, model, gen_type, masking)

            print(f"Sending request to {url}")

            response = requests.post(
                url,
                headers={
                    "Accept": "application/json" if self.is_img2img else None,
                    "Authorization": f"Bearer {manager.api_key}"
                },  
                files=files if self.is_img2img else None,
                data=payload if self.is_img2img else None,
                json=payload if not self.is_img2img else None,
            )


            assert response.status_code == 200, "Status Code: {} (expected {})".format(response.status_code, 200)
            
            response = response.json()
                
            images = response
            images = images["artifacts"]

            images = [PIL.Image.open(io.BytesIO(base64.b64decode(image["base64"]))) for image in images]
            images = [numpy.moveaxis(numpy.array(image).astype(numpy.float32) / 255.0, 2, 0) for image in images]
            images = [torch.from_numpy(image) for image in images]

            print(f"Returned {len(images)} image" + ("s" if len(images) > 1 else ''))

            return images                 
        
        except AssertionError:
            id = response.json()
            self.halt_process(p, id["message"])



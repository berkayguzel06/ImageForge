import transformers
import torch
import diffusers

class pipeline:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(pipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.pipe_type = "t2i"
        self.device_name = ""
        self.torch_dtype = ""
        self.model_path = "runwayml/stable-diffusion-v1-5"
        self.cache_dir = ".cache"
        self.clip_skip = 1
        self.pipe = ""
        self.text_encoder = ""
        self.scheduler = ""
        self.cuda()
        self.setup_system()
        self._set_scheduler()

    def cuda(self):
        if torch.cuda.is_available():
            print("Cuda is available")
            self.device_name = torch.device("cuda")
            self.torch_dtype = torch.float16
        else:
            self.device_name = torch.device("cpu")
            self.torch_dtype = torch.float32

    def _create_pipe(self):
        model_dir = None
        if self.model_path == "runwayml/stable-diffusion-v1-5":
            model_dir = self.cache_dir
        else:
            model_dir = None

        if self.pipe_type=="t2i":
            if self.clip_skip > 1:
                pipe = diffusers.DiffusionPipeline.from_pretrained(
                    self.model_path,
                    cache_dir=model_dir,
                    torch_dtype = self.torch_dtype,
                    safety_checker = None,
                    text_encoder = self.text_encoder,
                )
            else:
                pipe = diffusers.DiffusionPipeline.from_pretrained(
                    self.model_path,
                    cache_dir=model_dir,
                    torch_dtype = self.torch_dtype,
                    safety_checker = None
                )

        elif self.pipe_type=="i2i":
            if self.clip_skip > 1:
                pipe = diffusers.AutoPipelineForImage2Image.from_pretrained(
                    self.model_path,
                    cache_dir=model_dir,
                    torch_dtype = self.torch_dtype,
                    safety_checker = None,
                    text_encoder = self.text_encoder,
                )
            else:
                pipe = diffusers.AutoPipelineForImage2Image.from_pretrained(
                    self.model_path,
                    cache_dir=model_dir,
                    torch_dtype = self.torch_dtype,
                    safety_checker = None
                )
        return pipe
    
    def _create_text_encoder(self):
        if self.clip_skip > 1:
            text_encoder = transformers.CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir = self.cache_dir, subfolder = "text_encoder",
                                                                      num_hidden_layers = 12 - (self.clip_skip - 1),torch_dtype = self.torch_dtype)
        else:
            text_encoder = transformers.CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5",cache_dir = self.cache_dir,subfolder = "text_encoder",num_hidden_layers = 12,torch_dtype = self.torch_dtype)
        
        return text_encoder

    def _set_scheduler(self):
        self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def setup_system(self):
        self.text_encoder = self._create_text_encoder()
        self.pipe = self._create_pipe()
        self.pipe = self.pipe.to(self.device_name)

    def system_check(self, pipe_type, clip_skip, selected_model):
        check_flag = False

        if self.clip_skip!=clip_skip:
            self.clip_skip = clip_skip
            check_flag=True
        if self.pipe_type!=pipe_type:
            self.pipe_type = pipe_type
            check_flag=True
        if selected_model!=self.model_path:
            self.model_path = selected_model
            check_flag=True

        if check_flag:
            self.setup_system()

    def get_pipe(self):
        return self.pipe
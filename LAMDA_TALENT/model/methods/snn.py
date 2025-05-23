from model.methods.base import Method

class SNNMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from model.models.snn import SNN
        self.model = SNN(
                categories=self.categories,
                d_in=self.d_in,
                d_out=self.d_out,
                **model_config
                ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

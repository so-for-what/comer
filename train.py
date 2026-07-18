
from pytorch_lightning.cli import LightningCLI

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
import torch
torch.set_float32_matmul_precision('medium')  # 建议设置的精度，考虑到第一次跑的结果并不好，把它注释掉
def main():
    #torch.set_float32_matmul_precision('medium')  # 建议设置的精度
    
    cli = LightningCLI(
        LitCoMER,
        CROHMEDatamodule,
        save_config_kwargs={'overwrite': True},  # 已修正旧写法
        trainer_defaults={},
    )
    # 注意：LightningCLI 实例化后会自动运行子命令（fit/validate等），不需要额外调用

if __name__ == '__main__':
    main()

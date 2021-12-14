import torch
from mynet import *
from data_loader import get_loader
import time
from functools import wraps
from tqdm import tqdm


def time_fn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@time_fn: {fn.__name__} took {t2 - t1: .5f} s")
        return result

    return measure_time


@time_fn
def test(model, dataLoader):
    batch_tail = 0
    test_correct = 0
    for i, (images, labels) in enumerate(tqdm(dataLoader)):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        test_correct += torch.sum(preds == labels.data).to(torch.float32)
        batch_tail += len(labels)
    test_acc = (test_correct / batch_tail).item()
    print(model.__class__.__name__, " 测试准确率为: ", test_acc, end='\t')


# ========= 测试完整模型 =========
test_model_full = ResRepNet()
test_model_full.load_state_dict(torch.load("./mynet.pth"))
test_model_full.eval()
test_model_full.cuda()
test(test_model_full, get_loader()[1])


# ========= 测试等价转换后的模型 =========
test_model_rep = ResRepNet()
test_model_rep.load_state_dict(torch.load("./mynet.pth"))
for m in test_model_rep.modules():  # 等价转换
    if hasattr(m, 'switch_to_deploy'):
        m.switch_to_deploy()
print(test_model_rep)
test_model_rep.eval()
test_model_rep.cuda()
test(test_model_rep, get_loader()[1])

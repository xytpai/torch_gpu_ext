import torch
import torch_gpu_ext


myobj = torch_gpu_ext.MyObject(42)
print(myobj.value())


def test_gpu_add():
    a = torch.rand(5).cuda()
    b = torch.rand(5).cuda()
    ref_ab = a + b
    out_ab = torch.empty_like(a)
    torch.ops.torch_gpu_ext.gpu_add(a, b, out_ab)
    print("ref_add_out:", ref_ab)
    print("add_out:", out_ab)
    assert torch.allclose(ref_ab, out_ab)


def test_gpu_mul():
    a = torch.rand(5).cuda()
    b = torch.rand(5).cuda()
    ref_ab = a * b
    out_ab = torch.empty_like(a)
    torch.ops.torch_gpu_ext.gpu_mul(a, b, out_ab)
    print("ref_mul_out:", ref_ab)
    print("mul_out:", out_ab)
    assert torch.allclose(ref_ab, out_ab)


if __name__ == "__main__":
    test_gpu_add()
    test_gpu_mul()

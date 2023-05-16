import torch



def to_index(style):
    if style == "de":
        return 0#[1,0,0,0]
    elif style == "di":
        return 1#[0,1,0,0]
    elif style == "me":
        return 2#[0,0,1,0]
    elif style == "mi":
        return 3#[0,0,0,1]
    else:
        raise ValueError

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate(batch):
    databatch = [b[2] for b in batch]
    filebatch = [b[1] for b in batch]
    stylebatch = [to_index(b[0]) for b in batch]
    lenbatch = [len(b[2]) for b in batch]

    databatchTensor = collate_tensors(databatch)
    targetbatch = collate_tensors(databatch)
    #acntionbatchTensor = torch.as_tensor(actionbatch)
    filebatchTensor = torch.as_tensor(filebatch)
    stylebatchTensor = torch.as_tensor(stylebatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)

    batch = {"x": databatchTensor,"style":stylebatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor,"file": filebatchTensor}
    return batch

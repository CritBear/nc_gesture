
def process_file_name(file_name):
    d = file_name.split('_')
    action =''.join(d[:4])
    return action

def get_style_from_name(file_name):
    d = file_name.split('_')
    return d[-2]

def to_style_onehot_label(file_name):
    style = get_style_from_name(file_name)
    return to_onehot_label(style)

def get_onehot_labels(styles):
    labels = []
    for s in styles:
        labels.append(to_style_onehot_label(s))
    return labels

def to_onehot_label(style):
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


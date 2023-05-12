
def process_file_name(file_name):
    d = file_name.split('_')
    action =''.join(d[:4])
    return action

def get_style_from_name(file_name):
    d = file_name.split('_')
    return d[-2]

def to_style_index(file_name):
    style = get_style_from_name(file_name)
    return to_index(style)

def get_onehot_labels(styles):
    labels = []
    for s in styles:
        labels.append(to_style_index(s))
    return labels

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


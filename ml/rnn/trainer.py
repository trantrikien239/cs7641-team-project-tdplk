import torch

def train_grader(train_dl, valid_dl, model, loss_func, 
    opt, epochs=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for (xb, sb), yb in train_dl:
            xb = xb.to(device)
            sb = sb.to(device)
            yb = yb.to(device)
            # print(yb.dtype)
            yb_pred = model([xb, sb])
            loss = loss_func(yb_pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        with torch.no_grad():
            valid_loss = sum(
                loss_func(model(xb.to(device), sb.to(device)),
                    yb.to(device)) for ([xb, sb]), yb in valid_dl)
        print(epoch, valid_loss / len(valid_dl))
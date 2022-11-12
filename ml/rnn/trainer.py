import torch
from copy import deepcopy

def train_grader(train_dl, valid_dl, model, model_name, loss_func, 
    opt, save_path=".", epochs=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_eval_loss = float('inf')
    eval_loss_list = []
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
            train_loss = sum(
                loss_func(model([xb.to(device), sb.to(device)]),
                    yb.to(device)) for ([xb, sb]), yb in train_dl
            )
            valid_loss = sum(
                loss_func(model([xb.to(device), sb.to(device)]),
                    yb.to(device)) for ([xb, sb]), yb in valid_dl)
        tl = (train_loss / len(train_dl)).cpu().numpy()
        vl = (valid_loss / len(valid_dl)).cpu().numpy()
        if vl < best_eval_loss and vl > tl:
            best_eval_loss = vl
            best_model = deepcopy(model)
            torch.save(model.state_dict(), f'{save_path}/best_{model_name}.pt')
        eval_loss_list.append(vl)
        print(f"Epoch {epoch:04}, train loss: {tl:.6f}, valid loss: {vl:.6f}, best valid loss: {best_eval_loss:.6f}")
        if epoch > 5 and min(eval_loss_list[-10:]) > best_eval_loss:
            print(f"Early stopping, best valid loss: {best_eval_loss:.6f}")
            break
    # best_model = torch.load(f'best_model_{model_name}.pt')
    best_model.eval()
    return best_model
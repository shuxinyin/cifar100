import torch


def model_eval(model, test_dataloder, test_acc_list):
    model.eval()
    total_num, total_correct = 0, 0
    for i, (image, label) in enumerate(test_dataloder):
        image, label = image.cuda(), label.cuda()
        model_out = model(image)
        pred = torch.argmax(model_out, dim=1)
        correct = (pred == label).sum()
        total_correct += correct
        #     print(model_out.size(),label.size(), image.size())
        total_num += label.size(0)
        acc_tmp = int(total_correct) / total_num
        test_acc_list.append(acc_tmp)
    print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * acc_tmp))

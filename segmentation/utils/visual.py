"""
To visualize the segmentation results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

import skimage
from skimage import io as skio
import SimpleITK as sitk
from matplotlib.pyplot import GridSpec

##### Select the target cases
image_dir = './raw_data'
predict_dir = './output_bs32/dg'
color_list = ['green', 'purple', 'yellow']
linewidth = 1.

def select_cases(predict_dir, methods, domain_list:list=None):
    compare_methods = methods[:-1]
    target_method = methods[-1]

    metric = "Dice"
    # domain_list = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
    if domain_list is None:
        domain_list = [name for name in os.listdir(os.path.join(predict_dir, methods[0])) \
                            if not os.path.isfile(os.path.join(predict_dir,methods[0], name))]

    domain_select = {}

    for domain in domain_list:
        domain_results = {}
        domain_case_results = {}
        for method in methods:
            temp_summary_dir = os.path.join(predict_dir, method, domain, "summary")
            with open(os.path.join(temp_summary_dir, domain+"_detail_result.json")) as f:
                temp_summary = json.load(f)['case_level']
            domain_results[method] = temp_summary
        
        for case_name in list(temp_summary.keys()):
            compare_value = 1
            label_names = list(domain_results[target_method][case_name].keys())
            label_names.remove('Background')
            label_names.remove('case_name')

            for method in compare_methods:
                for label_name in label_names:
                    compare_value = min(compare_value, \
                                        domain_results[target_method][case_name][label_name][metric]-\
                                        domain_results[method][case_name][label_name][metric])

            if compare_value > 0:
                domain_case_results[case_name] = compare_value

        domain_select[domain] = domain_case_results

    best_domain_select = {}

    for domain in list(domain_list):
        best_index = np.argmax(np.array(list(domain_select[domain].values())))
        best_domain_select[domain] = list(domain_select[domain].keys())[best_index]
    
    return domain_select, best_domain_select


def visual_2d(image_dir, predict_dir, out_image_dir, 
              num_classes, methods, domain_list, best_domain_select):
    n_domains = len(domain_list)
    n_methods = len(methods)


    plt.figure(figsize=(2*(n_methods+1-0.25), 2*(n_domains-0.25)))
    gs = GridSpec(
            nrows=n_domains, ncols=(n_methods+1),
            left=0., bottom=0., right=1., top=1.,
            wspace=0.05, hspace=0.05)

    for index_domain, domain in enumerate(domain_list):
        image = skio.imread(os.path.join(image_dir, domain, "imagesTr", best_domain_select[domain]+'.png'))
        label = skio.imread(os.path.join(predict_dir, methods[0], domain, f"out_image/{domain}/label", best_domain_select[domain]+'.png'))
        label = ((label / np.max(label))*num_classes).astype(np.uint8)
        ax = plt.subplot(gs[index_domain, 0])
        ax.imshow(image)
        for class_index in range(num_classes):
            plt.contour(predict==(class_index+1), colors=color_list[class_index], linewidths=linewidth)
        ax.grid(False)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

        for index_method, method in enumerate(methods):
            predict = skio.imread(os.path.join(predict_dir, method, domain, f"out_image/{domain}/prediction", best_domain_select[domain]+'.png'))
            predict = ((predict / np.max(predict))*num_classes).astype(np.uint8)
            ax = plt.subplot(gs[index_domain, index_method+1])
            ax.imshow(image)

            if index_domain == 0:
                pass
                # ax.set_title(method)
            for class_index in range(num_classes):
                plt.contour(predict==(class_index+1), colors=color_list[class_index], linewidths=linewidth)

            ax.grid(False)
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_image_dir)


def visual_3d(image_dir, predict_dir, out_image_dir, 
              num_classes, methods, domain_list, best_domain_select):
    n_domains = len(domain_list)
    n_methods = len(methods)

    plt.figure(figsize=(2*(n_methods+1-0.25), 2*(n_domains-0.25)))
    gs = GridSpec(
            nrows=n_domains, ncols=(n_methods+1),
            left=0., bottom=0., right=1., top=1.,
            wspace=0.05, hspace=0.05)

    for index_domain, domain in enumerate(domain_list):
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_dir, domain, "imagesTr", best_domain_select[domain]+'.nii.gz')))
        label = sitk.GetArrayFromImage(sitk.ReadImage((os.path.join(predict_dir, methods[0], domain, \
                                                                    f"out_image/{domain}/label", best_domain_select[domain]+'.nii.gz'))))
        # label = ((label / np.max(label))*len(label_names)).astype(np.uint8)
        depth_index = np.argmax(np.sum(label, axis=(1, 2)))-1
        image = image[depth_index]
        label = label[depth_index]

        ax = plt.subplot(gs[index_domain, 0])
        ax.imshow(image, cmap='gray')

        for class_index in range(num_classes):
            plt.contour(predict==(class_index+1), colors=color_list[class_index], linewidths=linewidth)
        ax.grid(False)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

        for index_method, method in enumerate(methods):
            predict = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(predict_dir, method, domain, f"out_image/{domain}/prediction", best_domain_select[domain]+'.nii.gz')))
            predict = predict[depth_index]
            # predict = ((predict / np.max(predict))*len(label_names)).astype(np.uint8)
            ax = plt.subplot(gs[index_domain, index_method+1])
            ax.imshow(image, cmap='gray')

            if index_domain == 0:
                pass
                # ax.set_title(method)
            for class_index in range(num_classes):
                plt.contour(predict==(class_index+1), colors=color_list[class_index], linewidths=linewidth)

            ax.grid(False)
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_image_dir)

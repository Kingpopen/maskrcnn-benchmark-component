from .coco_eval import do_coco_evaluation as do_orig_coco_evaluation
from .coco_eval_wrapper import do_coco_evaluation as do_wrapped_coco_evaluation
from maskrcnn_benchmark.data.datasets import AbstractDataset, COCODataset


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    # 如果是coco数据集类型 直接进行coco的测评
    if isinstance(dataset, COCODataset):
        return do_orig_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )

    # 如果是抽象类的类型（这个地方不太明白~）
    elif isinstance(dataset, AbstractDataset):
        return do_wrapped_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    else:
        raise NotImplementedError(
            (
                "Ground truth dataset is not a COCODataset, "
                "nor it is derived from AbstractDataset: type(dataset)="
                "%s" % type(dataset)
            )
        )

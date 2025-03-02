from pycocotools.cocoeval import COCOeval

from ultralytics.utils import LOGGER
import numpy as np

class coco_eval(COCOeval):
    def summarize(self, filename):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            LOGGER.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return {
                'title': titleStr,
                'type': typeStr,
                'iou': iouStr,
                'area': areaRng,
                'maxDets': maxDets,
                'value': mean_s
            }, mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            details = [{} for _ in range(12)]
            details[0], stats[0] = _summarize(1)
            details[1], stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            details[2], stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            details[3], stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            details[4], stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            details[5], stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            details[6], stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            details[7], stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            details[8], stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            details[9], stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            details[10], stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            details[11], stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return details, stats

        def _summarizeKps():
            stats = np.zeros((10,))
            details = [{} for _ in range(10)]
            details[0], stats[0] = _summarize(1, maxDets=20)
            details[1], stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            details[2], stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            details[3], stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            details[4], stats[4] = _summarize(1, maxDets=20, areaRng='large')
            details[5], stats[5] = _summarize(0, maxDets=20)
            details[6], stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            details[7], stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            details[8], stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            details[9], stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return details, stats

        def save_to_csv(filename, details):
            '''
            将评估结果保存到CSV文件中
            '''
            import csv
            headers = ['Title', 'Type', 'IoU', 'Area', 'MaxDets', 'Value']
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for detail in details:
                    row = [
                        detail['title'],
                        detail['type'],
                        detail['iou'],
                        detail['area'],
                        detail['maxDets'],
                        detail['value']
                    ]
                    writer.writerow(row)

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        details, self.stats = summarize()
        save_to_csv(filename, details)
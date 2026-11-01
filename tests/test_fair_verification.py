import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fair_verification import relabel,disjoint,summary,scored,fourth_rows,fourth_schema,fourth_answer
class FairDataTests(unittest.TestCase):
    def row(self,source='export-review'):
        return dict(posting_id='a',employer_keys=['acme'],label_source=source,gold={'verdict':'stretch'},questions={})
    def test_resolved_negative(self):
        r=self.row();x=relabel(r);self.assertIs(x['gold']['needs_verification'],False);self.assertNotIn('needs_verification',r['gold'])
    def test_positive_no_verdict(self):
        x=relabel(self.row('export-weak-needs'));self.assertEqual(x['gold'],{'needs_verification':True});self.assertNotIn('verdict',x['questions'])
    def test_old_matched_export(self):
        x=relabel(self.row('v4'),{'verdict_normalized':'reject'});self.assertIs(x['gold']['needs_verification'],False)
    def test_missing_unchanged(self):
        self.assertNotIn('needs_verification',relabel(self.row('v4'))['gold'])
    def test_employer_overlap(self):
        a=self.row();b=dict(a,posting_id='b')
        with self.assertRaises(ValueError):disjoint([a],[b])
    def test_single_class_metrics(self):
        x=summary([dict(gold={'needs_verification':False},p=.8)],.5);self.assertIsNone(x['roc_auc']);self.assertEqual(x['false_positive_rate'],1)
    def test_nonreject_shortcut_fails_ranking(self):
        rows=[dict(gold={'needs_verification':y},p=.8) for y in [True,True,False,False]]
        self.assertEqual(summary(rows,.5)['roc_auc'],.5)
    def test_invalid_probability(self):
        r=self.row()
        with self.assertRaises(ValueError):scored([r],[dict(posting_id='a',probabilities={'needs_verification':{'true':float('nan')}})])
    def test_fourth_verdict_is_decoupled(self):
        r=relabel(self.row('export-weak-needs'));x=fourth_rows([r])[0]
        self.assertEqual(x['gold'],{'verdict':'needs_information'})
        self.assertEqual(r['gold'],{'needs_verification':True})
        self.assertNotIn('needs_verification',fourth_schema())
    def test_fourth_old_positive(self):
        r=self.row('old');r['gold']['needs_verification']=True
        self.assertEqual(fourth_rows([r])[0]['gold'],{'verdict':'needs_information'})
        self.assertNotIn('unresolved',fourth_schema()['verdict']['criteria']['stretch'])
    def test_fourth_resolved_keeps_verdict(self):
        x=fourth_rows([relabel(self.row())])[0]
        self.assertEqual(x['gold'],{'verdict':'stretch'})
    def test_fourth_serving_threshold(self):
        p={'needs_information':.4,'strong':.3,'stretch':.2,'reject':.1}
        self.assertEqual(fourth_answer(p,.4),'needs_information')
        self.assertEqual(fourth_answer(p,.5),'strong')
    def test_duplicate_predictions(self):
        p={'posting_id':'a'}
        with self.assertRaises(ValueError):scored([], [p,p])
if __name__=='__main__':unittest.main()

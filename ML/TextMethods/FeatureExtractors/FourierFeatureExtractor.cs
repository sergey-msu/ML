using System;

namespace ML.TextMethods.FeatureExtractors
{
  public class FourierFeatureExtractor: FeatureExtractorBase
  {
    public double T { get; set; }

    public override double[] ExtractFeatureVector(string doc, out bool isEmpty)
    {
      isEmpty    = true;
      var result = new double[DataDim];
      var reals  = new double[DataDim];
      var imags  = new double[DataDim];
      var tokens = Preprocessor.Preprocess(doc);
      if (tokens==null) return result;

      var jdx = -1;
      foreach (var token in tokens)
      {
        var idx = Vocabulary.IndexOf(token);
        if (idx<0) continue;

        jdx++;
        var jt = jdx*T;
        reals[idx]  += Math.Cos(jt);
        imags[idx]  += Math.Sin(jt);
        isEmpty = false;
      }

      for (int idx=0; idx<DataDim; idx++)
      {
        var r = reals[idx];
        var i = imags[idx];
        result[idx] = Math.Sqrt(r*r + i*i);
      }

      return result;
    }

    protected override void OnVocabularyChanged()
    {
      DataDim = Vocabulary.Count;
    }
  }
}

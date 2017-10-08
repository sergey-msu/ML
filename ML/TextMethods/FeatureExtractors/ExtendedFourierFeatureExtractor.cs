using System;

namespace ML.TextMethods.FeatureExtractors
{
  public class ExtendedFourierFeatureExtractor: FeatureExtractorBase
  {
    public double T { get; set; }

    public override double[] ExtractFeatureVector(string doc, out bool isEmpty)
    {
      isEmpty    = true;
      var dim    = Vocabulary.Count;
      var result = new double[DataDim];
      var reals  = new double[dim];
      var imags  = new double[dim];
      var tokens = Preprocessor.Preprocess(doc);
      if (tokens==null) return result;

      var jdx = -1;
      foreach (var token in tokens)
      {
        var idx = Vocabulary.IndexOf(token);
        if (idx<0) continue;

        jdx++;
        var jt = jdx*T;
        result[idx] += 1;
        reals[idx]  += Math.Cos(jt);
        imags[idx]  += Math.Sin(jt);
        isEmpty = false;
      }

      for (int idx=0; idx<dim; idx++)
      {
        var r = reals[idx];
        var i = imags[idx];
        result[dim+idx] = Math.Sqrt(r*r + i*i);
      }

      return result;
    }

    protected override void OnVocabularyChanged()
    {
      DataDim = Vocabulary.Count*2;
    }
  }
}

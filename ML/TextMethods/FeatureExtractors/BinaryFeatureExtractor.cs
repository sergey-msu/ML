using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.TextMethods.FeatureExtractors
{
  public class BinaryFeatureExtractor: FeatureExtractorBase
  {
    public override double[] ExtractFeatureVector(string doc, out bool isEmpty)
    {
      var result = new double[DataDim];
      var tokens = Preprocessor.Preprocess(doc);
      isEmpty = true;

      foreach (var token in tokens)
      {
        var idx = Vocabulary.IndexOf(token);
        if (idx<0) continue;
        result[idx] = 1;
        isEmpty = false;
      }

      return result;
    }

    protected override void OnVocabularyChanged()
    {
      DataDim = Vocabulary.Count;
    }
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Distributions;

namespace ML.TextMethods.Algorithms
{
  public class TFIDFNaiveBayesianAlgorithm : NaiveBayesianAlgorithmBase
  {
    private Dictionary<ClassFeatureKey, double> m_Frequencies;

    public TFIDFNaiveBayesianAlgorithm(ITextPreprocessor preprocessor)
      : base(preprocessor)
    {
    }

    #region Properties

    public override string ID   { get { return "TFIDFNB"; } }
    public override string Name { get { return "TF-IDF Naive Bayes Algorithm"; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      return null;
    }

    public override double[] ExtractFeatureVector(string doc)
    {
      return null;
    }


    protected override void TrainImpl()
    {
    }


  }
}

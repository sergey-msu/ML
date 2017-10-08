using System;
using System.Collections.Generic;
using ML.Core;
using ML.Contracts;
using ML.Core.Serialization;

namespace ML.TextMethods.Algorithms
{
  public class GeneralTextAlgorithm : TextAlgorithmBase
  {
    private IClassificationAlgorithm<double[]> m_SubAlgorithm;

    public GeneralTextAlgorithm(IClassificationAlgorithm<double[]> alg)
    {
      if (alg==null) throw new MLException("GeneralNaiveBayesianAlgorithm.ctor(alg=null)");

      m_SubAlgorithm = alg;
    }

    #region Properties

    public override string Name { get { return "GENNB"; } }

    public IClassificationAlgorithm<double[]> SubAlgorithm { get { return m_SubAlgorithm; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      bool isEmpty;
      var data = ExtractFeatureVector(obj, out isEmpty);
      return m_SubAlgorithm.PredictTokens(data, cnt);
    }


    protected override void TrainImpl()
    {
      var featureSample = new ClassifiedSample<double[]>();

      foreach (var pData in TrainingSample)
      {
        var doc  = pData.Key;
        bool isEmpty;
        var data = ExtractFeatureVector(doc, out isEmpty);
        if (isEmpty) continue;

        var cls  = pData.Value;
        featureSample[data] = cls;
      }

      m_SubAlgorithm.Train(featureSample);
    }

    #region Serialization

    public override void Serialize(MLSerializer ser)
    {
      ser.Write("SUB_ALGORITHM", m_SubAlgorithm);
    }

    public override void Deserialize(MLSerializer ser)
    {
      m_SubAlgorithm = ser.ReadMLSerializable("SUB_ALGORITHM") as IClassificationAlgorithm<double[]>;
    }

    #endregion
  }
}

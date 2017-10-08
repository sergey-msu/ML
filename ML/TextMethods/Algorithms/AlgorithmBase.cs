using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;
using ML.Core.Serialization;
using ML.TextMethods.FeatureExtractors;

namespace ML.TextMethods.Algorithms
{
  public abstract class TextAlgorithmBase : ClassificationAlgorithmBase<string>
  {
    #region Fields

    private ITextPreprocessor m_Preprocessor;
    private List<string> m_Vocabulary;
    private double[] m_PriorProbs;
    private int[] m_ClassHist;
    private int   m_DataDim;
    private int   m_DataCount;
    private bool  m_UsePriors;

    protected ITextFeatureExtractor m_FeatureExtractor;

    #endregion

    protected TextAlgorithmBase()
    {
      m_UsePriors = true;
      FeatureExtractor = Registry.TextFeatureExtractor.Multinomial;
    }

    #region Properties

    public List<string> Vocabulary { get { return m_Vocabulary; } }

    public ITextFeatureExtractor FeatureExtractor
    {
      get { return m_FeatureExtractor; }
      set
      {
        if (value==null)
          throw new MLException("FeatureExtractor can not be null");

        m_FeatureExtractor = value;
      }
    }

    public ITextPreprocessor Preprocessor
    {
      get { return m_Preprocessor; }
      set
      {
        if (value==null)
          throw new MLException("Preprocessor can not be null");

        m_Preprocessor = value;
      }
    }

    /// <summary>
    /// Prior class logarithm probabilities
    /// </summary>
    public double[] PriorProbs { get { return m_PriorProbs; } }
    public int[]    ClassHist  { get { return m_ClassHist; } }
    public int      DataDim    { get { return m_DataDim; } }
    public int      DataCount  { get { return m_DataCount; } }

    /// <summary>
    /// If true, the algorithm takes prior class probabilities into account
    /// </summary>
    public bool UsePriors { get { return m_UsePriors; } set { m_UsePriors=value; } }

    #endregion

    public virtual double[] ExtractFeatureVector(string doc, out bool isEmpty)
    {
      return FeatureExtractor.ExtractFeatureVector(doc, out isEmpty);
    }

    public virtual List<string> ExtractVocabulary(ClassifiedSample<string> corpus)
    {
      var dict = new HashSet<string>();

      foreach (var doc in corpus)
      {
        var tokens = m_Preprocessor.Preprocess(doc.Key);
        foreach (var token in tokens)
          dict.Add(token);
      }

      return dict.ToList();
    }


    protected override void DoTrain()
    {
      base.DoTrain();

      var classes = Classes.ToList();
      for (int i=0; i<classes.Count; i++)
      {
        var any = classes.Any(c => (int)c.Value==i);
        if (!any)  throw new MLException(string.Format("Class values must be enumerated from 0 to {0}", classes.Count));
      }

      var corpus = TrainingSample;
      if (corpus==null || !corpus.Any())
        throw new MLException("Training sample is null or empty");

      m_Vocabulary = ExtractVocabulary(corpus);
      if (m_Vocabulary.Count<=0)
        throw new MLException("Vocabulary is empty");

      m_FeatureExtractor.Preprocessor = m_Preprocessor;
      m_FeatureExtractor.Vocabulary = m_Vocabulary;
      m_ClassHist  = new int[classes.Count];
      m_PriorProbs = new double[classes.Count];
      m_DataDim    = FeatureExtractor.DataDim;
      m_DataCount  = TrainingSample.Count;

      foreach (var doc in TrainingSample)
      {
        var cls = doc.Value;
        m_ClassHist[cls.Value] += 1;
      }

      foreach (var cls in classes)
        m_PriorProbs[cls.Value] = Math.Log(m_ClassHist[cls.Value]/(double)m_DataCount);

      TrainImpl();
    }

    protected abstract void TrainImpl();

    #region Serialization

    public override void Serialize(MLSerializer ser)
    {
      base.Serialize(ser);

      ser.Write("FEATURE_EXTRACTOR", m_FeatureExtractor);
      ser.Write("PREPROCESSOR", m_Preprocessor);
      ser.Write("VOCABULARY",   m_Vocabulary);
      ser.Write("PRIORS",       m_PriorProbs);
      ser.Write("CLASS_HIST",   m_ClassHist);
      ser.Write("DATA_DIM",     m_DataDim);
      ser.Write("DATA_COUNT",   m_DataCount);
      ser.Write("USE_PRIORS",   m_UsePriors);
    }

    public override void Deserialize(MLSerializer ser)
    {
      base.Deserialize(ser);

      m_Preprocessor = ser.ReadObject<ITextPreprocessor>("PREPROCESSOR");
      m_Vocabulary   = ser.ReadStrings("VOCABULARY").ToList();
      m_FeatureExtractor = ser.ReadObject<ITextFeatureExtractor>("FEATURE_EXTRACTOR");
      m_FeatureExtractor.Preprocessor = m_Preprocessor;
      m_FeatureExtractor.Vocabulary = m_Vocabulary;
      m_PriorProbs = ser.ReadDoubles("PRIORS").ToArray();
      m_ClassHist  = ser.ReadInts("CLASS_HIST").ToArray();
      m_DataDim    = ser.ReadInt("DATA_DIM");
      m_DataCount  = ser.ReadInt("DATA_COUNT");
      m_UsePriors  = ser.ReadBool("USE_PRIORS");
    }

    #endregion
  }

  public abstract class NaiveBayesianAlgorithmBase : TextAlgorithmBase
  {
    private double[][] m_Weights;
    private double     m_Alpha;
    private bool       m_UseSmoothing;

    protected NaiveBayesianAlgorithmBase()
    {
      m_UseSmoothing = true;
      m_Alpha = 1;
    }

    #region Properties

    public double[][] Weights { get { return m_Weights; } }

    /// <summary>
    /// Smoothing coefficient
    /// </summary>
    public double Alpha
    {
      get { return m_Alpha; }
      set
      {
        if (value<=0)
          throw new MLException("Smoothing coefficient must be positive");

        m_Alpha=value;
      }
    }

    /// <summary>
    /// If true, uses Laplace/Lidstone smoothing
    /// </summary>
    public bool UseSmoothing { get { return m_UseSmoothing; } set { m_UseSmoothing=value; } }

    #endregion

    public override ClassScore[] PredictTokens(string obj, int cnt)
    {
      bool isEmpty;
      var data      = ExtractFeatureVector(obj, out isEmpty);
      var classes   = Classes;
      var priors    = PriorProbs;
      var dim       = DataDim;
      var usePriors = UsePriors;
      var scores    = new List<ClassScore>();

      foreach (var cls in classes)
      {
        var score = usePriors ? priors[cls.Value] : 0.0D;
        var weights = m_Weights[cls.Value];

        if (!isEmpty)
        {
          for (int i=0; i<dim; i++)
          {
            var x = data[i];
            if (x==0) continue;
            var w = weights[i];
            score += x*w;
          }
        }

        scores.Add(new ClassScore(cls, score));
      }

      return scores.OrderByDescending(s => s.Score)
                   .Take(cnt)
                   .ToArray();
    }


    protected override void TrainImpl()
    {
      m_Weights = TrainWeights();
    }

    protected abstract double[][] TrainWeights();

    #region Serialization

    public override void Serialize(MLSerializer ser)
    {
      base.Serialize(ser);

      ser.Write("USE_SMOOTHING", m_UseSmoothing);
      ser.Write("ALPHA", m_Alpha);
      ser.Write("WEIGHTS", m_Weights);
    }

    public override void Deserialize(MLSerializer ser)
    {
      base.Deserialize(ser);

      m_UseSmoothing = ser.ReadBool("USE_SMOOTHING");
      m_Alpha = ser.ReadDouble("ALPHA");
      m_Weights = ser.ReadObject<double[][]>("WEIGHTS");
    }

    #endregion
  }
}

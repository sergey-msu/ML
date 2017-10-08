using System;
using System.Collections.Generic;
using ML.Core;
using ML.Contracts;

namespace ML.TextMethods.FeatureExtractors
{
  public abstract class FeatureExtractorBase: ITextFeatureExtractor
  {
    [NonSerialized] private ITextPreprocessor m_Preprocessor;
    [NonSerialized] private List<string> m_Vocabulary;

    public FeatureExtractorBase()
    {
    }


    public ITextPreprocessor Preprocessor
    {
      get { return m_Preprocessor; }
      set
      {
        if (value==null)
          throw new MLException("Preprocessor can not be null)");

        m_Preprocessor = value;
        OnPreprocessorChanged();
      }
    }

    public List<string> Vocabulary
    {
      get { return m_Vocabulary; }
      set
      {
        if (value==null)
          throw new MLException("Vocabulary can not be null)");

        m_Vocabulary = value;
        OnVocabularyChanged();
      }
    }

    public int DataDim { get; protected set; }

    public abstract double[] ExtractFeatureVector(string doc, out bool isEmpty);


    protected virtual void OnVocabularyChanged()
    {
    }
    protected virtual void OnPreprocessorChanged()
    {
    }
  }
}

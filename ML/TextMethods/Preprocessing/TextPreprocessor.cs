using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;

namespace ML.TextMethods.Preprocessing
{
  /// <summary>
  /// Facilitates all text document preprocessing logic from text string to final sequence of normalized tokens
  /// </summary>
  public class TextPreprocessor : ITextPreprocessor
  {
    #region Fields

    private readonly ITokenizer  m_Tokenizer;
    private readonly IStopwords  m_Stopwords;
    private readonly INormalizer m_Normalizer;
    private readonly IStemmer    m_Stemmer;

    #endregion

    public TextPreprocessor(ITokenizer tokenizer,
                            IStopwords stopwords,
                            INormalizer normalizer,
                            IStemmer stemmer)
    {
      if (tokenizer==null)  throw new MLException("TextPreprocessor.ctor(tokenizer=null)");
      if (stopwords==null)  throw new MLException("TextPreprocessor.ctor(stopwords=null)");
      if (normalizer==null) throw new MLException("TextPreprocessor.ctor(normalizer=null)");
      if (stemmer==null)    throw new MLException("TextPreprocessor.ctor(stemmer=null)");

      m_Tokenizer  = tokenizer;
      m_Stopwords  = stopwords;
      m_Normalizer = normalizer;
      m_Stemmer    = stemmer;
    }

    #region Properties

    public ITokenizer  Tokenizer  { get { return m_Tokenizer; } }
    public IStopwords  Stopwords  { get { return m_Stopwords; } }
    public INormalizer Normalizer { get { return m_Normalizer; } }
    public IStemmer    Stemmer    { get { return m_Stemmer; } }

    #endregion

    public virtual List<string> Preprocess(string doc)
    {
      if (string.IsNullOrWhiteSpace(doc)) return null;

      var result = new List<string>();

      var tokens = Tokenizer.Tokenize(doc);
      foreach (var token in tokens)
      {
        if (Stopwords.Contains(token)) continue;
        var normalized = Normalizer.Normalize(token);
        if (string.IsNullOrWhiteSpace(normalized)) continue;

        var stem = Stemmer.Stem(normalized);

        result.Add(stem);
      }

      return result;
    }

  }
}

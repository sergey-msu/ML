using ML.Core;
using System;
using System.Collections.Generic;

namespace ML.Contracts
{
  /// <summary>
  /// Facilitates all text document preprocessing logic from text string to final sequence of normalized tokens
  /// </summary>
  public interface ITextPreprocessor
  {
    /// <summary>
    /// Given text document returns sequence of normalized tokens
    /// (e.g. "A pioneer in the then-burgeoning world of commercial magazine fiction"
    ///        --> ["pioneer", "burgeon", "world", "commercial", "magazine", "fiction"])
    /// </summary>
    List<string> Preprocess(string doc);
  }

  /// <summary>
  /// Contract for base document tokenizer (e.g. splitting document into words by a space, regexp etc. deleting commas, digits etc.:
  /// "she cuts further, out jumped one after another in 1984"
  ///  --> ["she", "cut", "further", "out", "jumped", "one", "after", "another" "in" )
  /// </summary>
  public interface ITokenizer
  {
    /// <summary>
    /// Performs basic dovument tokenization
    /// </summary>
    List<string> Tokenize(string doc);
  }

  /// <summary>
  /// Contract for container for stopwords - the one that must not be taken into account (e.g. "the", "me", "a" etc.)
  /// </summary>
  public interface IStopwords
  {
    /// <summary>
    /// Returns all stopwords
    /// </summary>
    IEnumerable<string> All();

    /// <summary>
    /// Returns true if passed word is a stopword, false otherwise
    /// </summary>
    bool Contains(string word);

    /// <summary>
    /// Adds new stopword to known stopword collection
    /// </summary>
    void Add(string stopword);
  }

  /// <summary>
  /// Contract for token normalizer (e.g. "New-York" -> "newyork" etc.)
  /// </summary>
  public interface INormalizer
  {
    /// <summary>
    /// Normalizes given token
    /// </summary>
    string Normalize(string token);
  }

  /// <summary>
  /// Contract for stemmer algorithm that transforms word into its root form (e.g. "sses" -> "s", "ies" -> "i" etc.)
  /// </summary>
  public interface IStemmer
  {
    /// <summary>
    /// Returns stemmed token
    /// </summary>
    string Stem(string token);
  }

  //TODO: Lemmatization

  public interface ITFWeightingScheme : INamed
  {
    void Reset();

    double GetFrequency(double[] initFreqs, int idx);
  }

  public interface IIDFWeightingScheme : INamed
  {
    double[] GetWeights(int vocabularyCount, int[] idfFreqs);
  }
}

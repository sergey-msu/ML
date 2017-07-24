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
    string[] Preprocess(string doc);
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
    string[] Tokenize(string doc);
  }

  /// <summary>
  /// Contract for container for stopword - the one that must not be taken into account (e.g. "the", "me", "a" etc.)
  /// </summary>
  public interface IStopwordsContainer
  {
    /// <summary>
    /// Returns all stopwords
    /// </summary>
    IEnumerable<string> GetStopwords();

    /// <summary>
    /// Adds new stopword to known stopword collection
    /// </summary>
    void AddStopword(string stopword);
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
  /// Contract for stemmer algorithm (e.g. "sses" -> "s", "ies" -> "i" etc.)
  /// </summary>
  public interface IStemmer
  {
    /// <summary>
    /// Returns stemmed token
    /// </summary>
    string Stem(string token);
  }

  //TODO: Lemmatization
}

using System;
using System.Collections.Generic;
using System.Globalization;
using ML.Contracts;

namespace ML.TextMethods.Tokenization
{
  /// <summary>
  /// The simplest document tokenizer - splits document into words by a space, ignores empty entries
  /// </summary>
  public class EnglishSimpleTokenizer : ITokenizer
  {
    //#region Inner
    //
    //public enum DashMode
    //{
    //  /// <summary>
    //  /// Remove orphan dashes:
    //  /// self-explanatory   -> "self-explanatory"
    //  /// self- explanatory  -> "self", "explanatory"
    //  /// self -explanatory  -> "self", "explanatory"
    //  /// self - explanatory -> "self", "explanatory"
    //  /// </summary>
    //  None = 0,
    //
    //  /// <summary>
    //  /// Remove orphan dashes:
    //  /// self-explanatory   -> "self-explanatory"
    //  /// self- explanatory  -> "self-explanatory"
    //  /// self -explanatory  -> "self-explanatory"
    //  /// self - explanatory -> "self-explanatory"
    //  /// </summary>
    //  BindBothSizes = 1,
    //
    //  /// <summary>
    //  /// Remove orphan dashes:
    //  /// self-explanatory   -> "self-explanatory"
    //  /// self- explanatory  -> "self-explanatory"
    //  /// self -explanatory  -> "self-explanatory"
    //  /// self - explanatory -> "self", "explanatory"
    //  /// </summary>
    //  BindOneSize = 2
    //}
    //
    //#endregion

    public const string DEFAULT_SEPARATOR = " ";

    public EnglishSimpleTokenizer()
    {
      SplitSeparator = DEFAULT_SEPARATOR;
      RemoveNumbers = true;
      RemoveNonLetterMarks = true;
    }

    public string SplitSeparator { get; set; }
    public bool   RemoveNumbers { get; set; }
    public bool   RemoveNonLetterMarks { get; set; }

    public List<string> Tokenize(string doc)
    {
      if (string.IsNullOrWhiteSpace(doc)) return null;

      var result = new List<string>();
      var elems = doc.Split(new[] { SplitSeparator }, StringSplitOptions.RemoveEmptyEntries);
      var len = elems.Length;

      for (int i=0; i<len; i++)
      {
        var elem = elems[i];

        if (RemoveNonLetterMarks && doestNotContainsLetters(elem)) continue;
        if (RemoveNumbers)
        {
          double num;
          if (double.TryParse(elem, NumberStyles.Any, CultureInfo.InvariantCulture,  out num)) continue;
        }

        result.Add(elem);
      }

      return result;
    }

    #region .pvt

    private bool doestNotContainsLetters(string elem)
    {
      var len = elem.Length;
      for (int i=0; i<len; i++)
        if (char.IsLetter(elem[i]))return false;

      return true;
    }

    #endregion
  }
}

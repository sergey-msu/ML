using System;
using System.Collections.Generic;
using System.Text;
using ML.Contracts;

namespace ML.TextMethods.Normalization
{
  /// <summary>
  /// Contract for token normalizer (e.g. "New-York" -> "newyork" etc.)
  /// </summary>
  public class EnglishSimpleNormalizer : INormalizer
  {
    public EnglishSimpleNormalizer()
    {
      RemoveDashes = true;
      OnlyLettersAndDigits = true;
    }

    public bool RemoveDashes { get; set; }
    public bool OnlyLettersAndDigits { get; set; }
    public int  MinShortWordLength { get; set; }

    public string Normalize(string token)
    {
      token = token.ToLowerInvariant();
      var chars = token.ToCharArray();
      var len = chars.Length;
      var builder = new StringBuilder(len);

      for (int i=0; i<len; i++)
      {
        var c = chars[i];

        if (OnlyLettersAndDigits && !char.IsLetterOrDigit(c)) continue;
        if (RemoveDashes && c == '-') continue;

        builder.Append(c);
      }

      var result = builder.ToString();
      if (result.Length < MinShortWordLength) return null;

      return result;
    }
  }
}

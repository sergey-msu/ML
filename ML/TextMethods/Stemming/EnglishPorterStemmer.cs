using System;
using System.Collections.Generic;
using System.Text;
using ML.Contracts;

namespace ML.TextMethods.Stemming
{
  /// <summary>
	/// Porter stemming algorithm implementation
	/// </summary>
	public class EnglishPorterStemmer : IStemmer
  {
    #region Inner

    private class StemContext
    {
      public readonly StringBuilder Builder;
      public int EndIndex;
      public int StemIndex;

      public StemContext(StringBuilder builder, int stemIdx, int endIdx)
      {
        Builder = builder;
        EndIndex = endIdx;
        StemIndex = stemIdx;
      }
    }

    #endregion

    /// <summary>
    /// Returns stemmed token
    /// </summary>
    public string Stem(string token)
    {
      if (string.IsNullOrWhiteSpace(token) || token.Length <= 2) return token;

      var ctx = new StemContext(new StringBuilder(token), 0, token.Length-1);

      step1(ctx);
      step2(ctx);
      step3(ctx);
      step4(ctx);
      step5(ctx);
      step6(ctx);

      ctx.Builder.Length = ctx.EndIndex+1;

			return ctx.Builder.ToString();
    }

    #region .pvt

    /// <summary>
    /// Removes -ed, -ing and plurals
    /// e.g.
    ///   caresses  ->  caress
		///   ponies    ->  poni
    ///   matting   ->  mat
    ///   agreed    ->  agree
    /// </summary>
    private void step1(StemContext ctx)
    {
      if (ctx.Builder[ctx.EndIndex] == 's')
      {
             if (endsWith(ctx, "sses")) ctx.EndIndex -= 2;
        else if (endsWith(ctx, "ies")) setEnd(ctx, "i");
        else if (ctx.Builder[ctx.EndIndex - 1] != 's')
          ctx.EndIndex--;
      }
      if (endsWith(ctx, "eed"))
      {
        if (measureConsontantSequence(ctx) > 0)
          ctx.EndIndex--;
      }
      else if ((endsWith(ctx, "ed") || endsWith(ctx, "ing")) && vowelInStem(ctx))
      {
        ctx.EndIndex = ctx.StemIndex;
        if (endsWith(ctx, "at"))
          setEnd(ctx, "ate");
        else if (endsWith(ctx, "bl"))
          setEnd(ctx, "ble");
        else if (endsWith(ctx, "iz"))
          setEnd(ctx, "ize");
        else if (isDoubleConsontant(ctx, ctx.EndIndex))
        {
          ctx.EndIndex--;
          int ch = ctx.Builder[ctx.EndIndex];
          if (ch == 'l' || ch == 's' || ch == 'z')
            ctx.EndIndex++;
        }
        else if (measureConsontantSequence(ctx) == 1 && isCVC(ctx, ctx.EndIndex)) setEnd(ctx, "e");
      }
    }

    /// <summary>
    /// Change terminal y to i when there is another vowel in the stem
    /// </summary>
    private void step2(StemContext ctx)
    {
      if (endsWith(ctx, "y") && vowelInStem(ctx))
        ctx.Builder[ctx.EndIndex] = 'i';
    }

    /// <summary>
    /// Maps double suffices to single ones. so -ization ( = -ize plus -ation) maps to -ize etc.
    /// Note that the string before the suffix must give m() > 0.
    /// </summary>
    private void step3(StemContext ctx)
    {
      if (ctx.EndIndex == 0) return;

      switch (ctx.Builder[ctx.EndIndex-1])
      {
        case 'a':
               if (endsWith(ctx, "ational")) replaceEndIfConsonant(ctx, "ate");
          else if (endsWith(ctx, "tional"))  replaceEndIfConsonant(ctx, "tion");
          break;
        case 'c':
               if (endsWith(ctx, "enci")) replaceEndIfConsonant(ctx, "ence");
          else if (endsWith(ctx, "anci")) replaceEndIfConsonant(ctx, "ance");
          break;
        case 'e':
          if (endsWith(ctx, "izer")) replaceEndIfConsonant(ctx, "ize");
          break;
        case 'l':
               if (endsWith(ctx, "bli"))   replaceEndIfConsonant(ctx, "ble");
          else if (endsWith(ctx, "alli"))  replaceEndIfConsonant(ctx, "al");
          else if (endsWith(ctx, "entli")) replaceEndIfConsonant(ctx, "ent");
          else if (endsWith(ctx, "eli"))   replaceEndIfConsonant(ctx, "e");
          else if (endsWith(ctx, "ousli")) replaceEndIfConsonant(ctx, "ous");
          break;
        case 'o':
               if (endsWith(ctx, "ization")) replaceEndIfConsonant(ctx, "ize");
          else if (endsWith(ctx, "ation"))   replaceEndIfConsonant(ctx, "ate");
          else if (endsWith(ctx, "ator"))    replaceEndIfConsonant(ctx, "ate");
          break;
        case 's':
               if (endsWith(ctx, "alism"))   replaceEndIfConsonant(ctx, "al");
          else if (endsWith(ctx, "iveness")) replaceEndIfConsonant(ctx, "ive");
          else if (endsWith(ctx, "fulness")) replaceEndIfConsonant(ctx, "ful");
          else if (endsWith(ctx, "ousness")) replaceEndIfConsonant(ctx, "ous");
          break;
        case 't':
               if (endsWith(ctx, "aliti"))  replaceEndIfConsonant(ctx, "al");
          else if (endsWith(ctx, "iviti"))  replaceEndIfConsonant(ctx, "ive");
          else if (endsWith(ctx, "biliti")) replaceEndIfConsonant(ctx, "ble");
          break;
        case 'g':
          if (endsWith(ctx, "logi")) replaceEndIfConsonant(ctx, "log");
          break;
      }
    }

    /// <summary>
    /// Deals with -ic-, -full, -ness etc. similar strategy to step3.
    /// </summary>
    private void step4(StemContext ctx)
    {
      switch (ctx.Builder[ctx.EndIndex])
      {
        case 'e':
               if (endsWith(ctx, "icate")) replaceEndIfConsonant(ctx, "ic");
          else if (endsWith(ctx, "ative")) replaceEndIfConsonant(ctx, string.Empty);
          else if (endsWith(ctx, "alize")) replaceEndIfConsonant(ctx, "al");
          break;
        case 'i':
          if (endsWith(ctx, "iciti")) { replaceEndIfConsonant(ctx, "ic"); }
          break;
        case 'l':
               if (endsWith(ctx, "ical")) replaceEndIfConsonant(ctx, "ic");
          else if (endsWith(ctx, "ful"))  replaceEndIfConsonant(ctx, string.Empty);
          break;
        case 's':
          if (endsWith(ctx, "ness")) replaceEndIfConsonant(ctx, string.Empty);
          break;
      }
    }

    /// <summary>
    /// Removes -ence, -ant etc., in context <c>vcvc<v>
    /// </summary>
    private void step5(StemContext ctx)
    {
      if (ctx.EndIndex == 0) return;

      switch (ctx.Builder[ctx.EndIndex-1])
      {
        case 'a':
          if (endsWith(ctx, "al")) break; return;
        case 'c':
          if (endsWith(ctx, "ance")) break;
          if (endsWith(ctx, "ence")) break; return;
        case 'e':
          if (endsWith(ctx, "er")) break; return;
        case 'i':
          if (endsWith(ctx, "ic")) break; return;
        case 'l':
          if (endsWith(ctx, "able")) break;
          if (endsWith(ctx, "ible")) break; return;
        case 'n':
          if (endsWith(ctx, "ant")) break;
          if (endsWith(ctx, "ement")) break;
          if (endsWith(ctx, "ment")) break;
          if (endsWith(ctx, "ent")) break; return;
        case 'o':
          if (endsWith(ctx, "ion") && ctx.StemIndex >= 0 && (ctx.Builder[ctx.StemIndex] == 's' || ctx.Builder[ctx.StemIndex] == 't')) break;
          if (endsWith(ctx, "ou")) break; return;
        case 's':
          if (endsWith(ctx, "ism")) break; return;
        case 't':
          if (endsWith(ctx, "ate")) break;
          if (endsWith(ctx, "iti")) break; return;
        case 'u':
          if (endsWith(ctx, "ous")) break; return;
        case 'v':
          if (endsWith(ctx, "ive")) break; return;
        case 'z':
          if (endsWith(ctx, "ize")) break; return;
        default:
          return;
      }

      if (measureConsontantSequence(ctx) > 1)
        ctx.EndIndex = ctx.StemIndex;
    }

    /// <summary>
    /// Removes -e ending if m() > 1
    /// </summary>
    private void step6(StemContext ctx)
    {
      ctx.StemIndex = ctx.EndIndex;

      if (ctx.Builder[ctx.EndIndex] == 'e')
      {
        var a = measureConsontantSequence(ctx);
        if (a > 1 || a == 1 && !isCVC(ctx, ctx.EndIndex - 1))
          ctx.EndIndex--;
      }
      if (ctx.Builder[ctx.EndIndex] == 'l' && isDoubleConsontant(ctx, ctx.EndIndex) && measureConsontantSequence(ctx) > 1)
        ctx.EndIndex--;
    }


    /// <summary>
    /// Returns true if the character at the specified index is a consonant.
    /// With special handling for 'y'.
    /// </summary>
    private bool isConsonant(StemContext ctx, int index)
    {
      var c = ctx.Builder[index];
      if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return false;
      return c != 'y' || (index == 0 || !isConsonant(ctx, index - 1));
    }

    /// <summary>
    /// m() measures the number of consonant sequences between 0 and j.
    /// if c is consonant sequence and v a vowel sequence, and <..> indicates arbitrary
		/// presence,
    ///
		///  <c><v>       gives 0
		///  <c>vc<v>     gives 1
		///  <c>vcvc<v>   gives 2
		///  <c>vcvcvc<v> gives 3
		///  ....
    /// </summary>
    private int measureConsontantSequence(StemContext ctx)
    {
      var n = 0;
      var index = 0;
      while (true)
      {
        if (index > ctx.StemIndex) return n;
        if (!isConsonant(ctx, index)) break; index++;
      }
      index++;
      while (true)
      {
        while (true)
        {
          if (index > ctx.StemIndex) return n;
          if (isConsonant(ctx, index)) break;
          index++;
        }
        index++;
        n++;
        while (true)
        {
          if (index > ctx.StemIndex) return n;
          if (!isConsonant(ctx, index)) break;
          index++;
        }
        index++;
      }
    }

    /// <summary>
    /// Return true if there is a vowel in the current stem (0 ... stemIndex)
    /// </summary>
    private bool vowelInStem(StemContext ctx)
    {
      int i;
      for (i = 0; i <= ctx.StemIndex; i++)
      {
        if (!isConsonant(ctx, i)) return true;
      }
      return false;
    }

    /// <summary>
    /// Returns true if the char at the specified index and the one preceeding it are the same consonants
    /// </summary>
    private bool isDoubleConsontant(StemContext ctx, int index)
    {
      if (index < 1) return false;
      return ctx.Builder[index] == ctx.Builder[index - 1] && isConsonant(ctx, index);
    }

    /// <summary>
    /// cvc(i) is true <=> i-2,i-1,i has the form consonant - vowel - consonant
		/// and also if the second c is not w,x or y. this is used when trying to
		/// restore an e at the end of a short word. e.g.
		///   cav(e), lov(e), hop(e), crim(e), but
		///   snow, box, tray.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    private bool isCVC(StemContext ctx, int index)
    {
      if (index < 2 || !isConsonant(ctx, index) || isConsonant(ctx, index - 1) || !isConsonant(ctx, index - 2)) return false;
      var c = ctx.Builder[index];
      return c != 'w' && c != 'x' && c != 'y';
    }

    /// <summary>
    /// Does the current word array end with the specified string
    /// </summary>
    private bool endsWith(StemContext ctx, string str)
    {
      var length = str.Length;
      var index = ctx.EndIndex - length + 1;
      if (index < 0) return false;

      for (var i=0; i<length; i++)
        if (ctx.Builder[index+i] != str[i]) return false;

      ctx.StemIndex = ctx.EndIndex - length;

      return true;
    }

    /// <summary>
    /// Set the end of the word to s.
		/// Starting at the current stem pointer and readjusting the end pointer
    /// </summary>
    private void setEnd(StemContext ctx, string str)
    {
      var length = str.Length;
      var index = ctx.StemIndex+1;
      for (var i=0; i<length; i++)
        ctx.Builder[index+i] = str[i];

      ctx.EndIndex = ctx.StemIndex + length;
    }

    /// <summary>
    /// Conditionally replace the end of the word
    /// </summary>
    private void replaceEndIfConsonant(StemContext ctx, string s)
    {
      if (measureConsontantSequence(ctx) > 0) setEnd(ctx, s);
    }

    #endregion
  }
}

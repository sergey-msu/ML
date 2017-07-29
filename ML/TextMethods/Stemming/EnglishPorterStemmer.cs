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
		private StringBuilder m_Builder;
		private int m_EndIndex;
		private int m_StemIndex;


    /// <summary>
    /// Returns stemmed token
    /// </summary>
    public string Stem(string token)
    {
      if (string.IsNullOrWhiteSpace(token) || token.Length <= 2) return token;

      m_Builder   = new StringBuilder(token);
			m_StemIndex = 0;
			m_EndIndex  = token.Length-1;

      step1();
      step2();
      step3();
      step4();
      step5();
      step6();

      m_Builder.Length = m_EndIndex+1;

			return m_Builder.ToString();
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
    private void step1()
    {
      if (m_Builder[m_EndIndex] == 's')
      {
             if (endsWith("sses")) m_EndIndex -= 2;
        else if (endsWith("ies")) setEnd("i");
        else if (m_Builder[m_EndIndex - 1] != 's')
          m_EndIndex--;
      }
      if (endsWith("eed"))
      {
        if (measureConsontantSequence() > 0)
          m_EndIndex--;
      }
      else if ((endsWith("ed") || endsWith("ing")) && vowelInStem())
      {
        m_EndIndex = m_StemIndex;
        if (endsWith("at"))
          setEnd("ate");
        else if (endsWith("bl"))
          setEnd("ble");
        else if (endsWith("iz"))
          setEnd("ize");
        else if (isDoubleConsontant(m_EndIndex))
        {
          m_EndIndex--;
          int ch = m_Builder[m_EndIndex];
          if (ch == 'l' || ch == 's' || ch == 'z')
            m_EndIndex++;
        }
        else if (measureConsontantSequence() == 1 && isCVC(m_EndIndex)) setEnd("e");
      }
    }

    /// <summary>
    /// Change terminal y to i when there is another vowel in the stem
    /// </summary>
    private void step2()
    {
      if (endsWith("y") && vowelInStem())
        m_Builder[m_EndIndex] = 'i';
    }

    /// <summary>
    /// Maps double suffices to single ones. so -ization ( = -ize plus -ation) maps to -ize etc.
    /// Note that the string before the suffix must give m() > 0.
    /// </summary>
    private void step3()
    {
      if (m_EndIndex == 0) return;

      switch (m_Builder[m_EndIndex-1])
      {
        case 'a':
               if (endsWith("ational")) replaceEndIfConsonant("ate");
          else if (endsWith("tional"))  replaceEndIfConsonant("tion");
          break;
        case 'c':
               if (endsWith("enci")) replaceEndIfConsonant("ence");
          else if (endsWith("anci")) replaceEndIfConsonant("ance");
          break;
        case 'e':
          if (endsWith("izer")) replaceEndIfConsonant("ize");
          break;
        case 'l':
               if (endsWith("bli"))   replaceEndIfConsonant("ble");
          else if (endsWith("alli"))  replaceEndIfConsonant("al");
          else if (endsWith("entli")) replaceEndIfConsonant("ent");
          else if (endsWith("eli"))   replaceEndIfConsonant("e");
          else if (endsWith("ousli")) replaceEndIfConsonant("ous");
          break;
        case 'o':
               if (endsWith("ization")) replaceEndIfConsonant("ize");
          else if (endsWith("ation"))   replaceEndIfConsonant("ate");
          else if (endsWith("ator"))    replaceEndIfConsonant("ate");
          break;
        case 's':
               if (endsWith("alism"))   replaceEndIfConsonant("al");
          else if (endsWith("iveness")) replaceEndIfConsonant("ive");
          else if (endsWith("fulness")) replaceEndIfConsonant("ful");
          else if (endsWith("ousness")) replaceEndIfConsonant("ous");
          break;
        case 't':
               if (endsWith("aliti"))  replaceEndIfConsonant("al");
          else if (endsWith("iviti"))  replaceEndIfConsonant("ive");
          else if (endsWith("biliti")) replaceEndIfConsonant("ble");
          break;
        case 'g':
          if (endsWith("logi")) replaceEndIfConsonant("log");
          break;
      }
    }

    /// <summary>
    /// Deals with -ic-, -full, -ness etc. similar strategy to step3.
    /// </summary>
    private void step4()
    {
      switch (m_Builder[m_EndIndex])
      {
        case 'e':
               if (endsWith("icate")) replaceEndIfConsonant("ic");
          else if (endsWith("ative")) replaceEndIfConsonant("");
          else if (endsWith("alize")) replaceEndIfConsonant("al");
          break;
        case 'i':
          if (endsWith("iciti")) { replaceEndIfConsonant("ic"); }
          break;
        case 'l':
               if (endsWith("ical")) replaceEndIfConsonant("ic");
          else if (endsWith("ful"))  replaceEndIfConsonant("");
          break;
        case 's':
          if (endsWith("ness")) replaceEndIfConsonant("");
          break;
      }
    }

    /// <summary>
    /// Removes -ence, -ant etc., in context <c>vcvc<v>
    /// </summary>
    private void step5()
    {
      if (m_EndIndex == 0) return;

      switch (m_Builder[m_EndIndex-1])
      {
        case 'a':
          if (endsWith("al")) break; return;
        case 'c':
          if (endsWith("ance")) break;
          if (endsWith("ence")) break; return;
        case 'e':
          if (endsWith("er")) break; return;
        case 'i':
          if (endsWith("ic")) break; return;
        case 'l':
          if (endsWith("able")) break;
          if (endsWith("ible")) break; return;
        case 'n':
          if (endsWith("ant")) break;
          if (endsWith("ement")) break;
          if (endsWith("ment")) break;
          if (endsWith("ent")) break; return;
        case 'o':
          if (endsWith("ion") && m_StemIndex >= 0 && (m_Builder[m_StemIndex] == 's' || m_Builder[m_StemIndex] == 't')) break;
          if (endsWith("ou")) break; return;
        case 's':
          if (endsWith("ism")) break; return;
        case 't':
          if (endsWith("ate")) break;
          if (endsWith("iti")) break; return;
        case 'u':
          if (endsWith("ous")) break; return;
        case 'v':
          if (endsWith("ive")) break; return;
        case 'z':
          if (endsWith("ize")) break; return;
        default:
          return;
      }

      if (measureConsontantSequence() > 1)
        m_EndIndex = m_StemIndex;
    }

    /// <summary>
    /// Removes -e ending if m() > 1
    /// </summary>
    private void step6()
    {
      m_StemIndex = m_EndIndex;

      if (m_Builder[m_EndIndex] == 'e')
      {
        var a = measureConsontantSequence();
        if (a > 1 || a == 1 && !isCVC(m_EndIndex - 1))
          m_EndIndex--;
      }
      if (m_Builder[m_EndIndex] == 'l' && isDoubleConsontant(m_EndIndex) && measureConsontantSequence() > 1)
        m_EndIndex--;
    }


    /// <summary>
    /// Returns true if the character at the specified index is a consonant.
    /// With special handling for 'y'.
    /// </summary>
    private bool isConsonant(int index)
    {
      var c = m_Builder[index];
      if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return false;
      return c != 'y' || (index == 0 || !isConsonant(index - 1));
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
    private int measureConsontantSequence()
    {
      var n = 0;
      var index = 0;
      while (true)
      {
        if (index > m_StemIndex) return n;
        if (!isConsonant(index)) break; index++;
      }
      index++;
      while (true)
      {
        while (true)
        {
          if (index > m_StemIndex) return n;
          if (isConsonant(index)) break;
          index++;
        }
        index++;
        n++;
        while (true)
        {
          if (index > m_StemIndex) return n;
          if (!isConsonant(index)) break;
          index++;
        }
        index++;
      }
    }

    /// <summary>
    /// Return true if there is a vowel in the current stem (0 ... stemIndex)
    /// </summary>
    private bool vowelInStem()
    {
      int i;
      for (i = 0; i <= m_StemIndex; i++)
      {
        if (!isConsonant(i)) return true;
      }
      return false;
    }

    /// <summary>
    /// Returns true if the char at the specified index and the one preceeding it are the same consonants
    /// </summary>
    private bool isDoubleConsontant(int index)
    {
      if (index < 1) return false;
      return m_Builder[index] == m_Builder[index - 1] && isConsonant(index);
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
    private bool isCVC(int index)
    {
      if (index < 2 || !isConsonant(index) || isConsonant(index - 1) || !isConsonant(index - 2)) return false;
      var c = m_Builder[index];
      return c != 'w' && c != 'x' && c != 'y';
    }

    /// <summary>
    /// Does the current word array end with the specified string
    /// </summary>
    private bool endsWith(string str)
    {
      var length = str.Length;
      var index = m_EndIndex - length + 1;
      if (index < 0) return false;

      for (var i=0; i<length; i++)
        if (m_Builder[index+i] != str[i]) return false;

      m_StemIndex = m_EndIndex - length;

      return true;
    }

    /// <summary>
    /// Set the end of the word to s.
		/// Starting at the current stem pointer and readjusting the end pointer
    /// </summary>
    private void setEnd(string str)
    {
      var length = str.Length;
      var index = m_StemIndex+1;
      for (var i=0; i<length; i++)
        m_Builder[index+i] = str[i];

      m_EndIndex = m_StemIndex + length;
    }

    /// <summary>
    /// Conditionally replace the end of the word
    /// </summary>
    private void replaceEndIfConsonant(string s)
    {
      if (measureConsontantSequence() > 0) setEnd(s);
    }

    #endregion
  }
}

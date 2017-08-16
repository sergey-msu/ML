using ML.Contracts;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace ML.Core.Serialization
{
  public class MLSerializer
  {
    #region Inner

    public enum SerializationMode
    {
      Serialization,
      Deserialization
    }

    #endregion

    public const string ICHAR = "∫";
    public const string DCHAR = "∂";
    public const string NULL  = ICHAR+ICHAR+"NULL";

    private readonly StreamWriter m_Writer;
    private readonly StreamReader m_Reader;

    public MLSerializer(StreamWriter writer)
    {
      if (writer==null) throw new MLException("MLSerializer.ctor(writer=null)");

      m_Writer = writer;
      Mode = SerializationMode.Serialization;
    }

    public MLSerializer(StreamReader reader)
    {
      if (reader==null) throw new MLException("MLSerializer.ctor(reader=null)");

      m_Reader = reader;
      Mode = SerializationMode.Deserialization;
    }


    public SerializationMode Mode { get; private set; }

    #region Read

    public T ReadObject<T>(string name)
    {
      var block = readBlock(name);

      using (var ms = new MemoryStream(Convert.FromBase64String(block)))
      {
        ms.Position = 0;

        var slim = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
        return (T)slim.Deserialize(ms);
      }
    }

    public string ReadString(string name)
    {
      return readBlock(name);
    }

    public IEnumerable<T> ReadEnumerable<T>(string name, Func<string, T> converter)
    {
      var list = readList(name);
      foreach (var item in list)
        yield return converter(item);
    }

    public IEnumerable<string> ReadStrings(string name)
    {
      return readList(name);
    }

    public IEnumerable<double> ReadDoubles(string name)
    {
      return ReadEnumerable(name, s => double.Parse(s));
    }

    public IEnumerable<int> ReadInts(string name)
    {
      return ReadEnumerable(name, s => int.Parse(s));
    }

    public int ReadInt(string name)
    {
      var val = readBlock(name);
      return int.Parse(val);
    }

    public double ReadDouble(string name)
    {
      var val = readBlock(name);
      return double.Parse(val);
    }

    public bool ReadBool(string name)
    {
      var val = readBlock(name);
      return bool.Parse(val);
    }

    public IMLSerializable ReadMLSerializable(string name)
    {
      throw new NotImplementedException();
    }

    #endregion

    #region Write

    public void Write(string name, IMLSerializable value)
    {
      write(name, value, v =>
      {
        value.Serialize(this);
      });
    }

    public void Write(string name, object value)
    {
      write(name, value, v =>
      {
        using (var ms = new MemoryStream())
        using (var reader = new BinaryReader(ms))
        {
          var slim = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
          slim.Serialize(ms, v);
          var len = ms.Position;
          ms.Position = 0;
          var str = Convert.ToBase64String(reader.ReadBytes((int)len), Base64FormattingOptions.None);

          m_Writer.WriteLine(str);
        }
      });
    }

    public void Write(string name, string value)
    {
      write(name, value, v =>
      {
        m_Writer.WriteLine(v);
      });
    }

    public void Write(string name, int value)
    {
      write(name, value, v =>
      {
        m_Writer.WriteLine(v);
      });
    }

    public void Write(string name, double value)
    {
      write(name, value, v =>
      {
        m_Writer.WriteLine("{0:0.000000}", v);
      });
    }

    public void Write(string name, bool value)
    {
      write(name, value, v =>
      {
        m_Writer.WriteLine(v);
      });
    }

    public void Write(string name, double[] value)
    {
      write(name, value, v =>
      {
        var len = v.Length;
        for (int i=0; i<len; i++)
          m_Writer.WriteLine("{0:0.000000}", v[i]);
      });
    }

    public void Write(string name, int[] value)
    {
      write(name, value, v =>
      {
        var len = v.Length;
        for (int i=0; i<len; i++)
          m_Writer.WriteLine("{0}", v[i]);
      });
    }

    public void Write(string name, IEnumerable<string> value)
    {
      write(name, value, v =>
      {
        foreach (var item in v)
          m_Writer.WriteLine(item);
      });
    }

    #endregion

    #region .pvt

    private void write<T>(string name, T value, Action<T> body)
    {
      if (Mode != SerializationMode.Serialization)
        throw new MLException("Incorrect serialization mode");

      writeHeader(name);

      if (value==null) m_Writer.Write(NULL);
      else body(value);

      writeFooter(name);
    }

    private void writeHeader(string name)
    {
      var header = string.Format("{0}{1}{0}", ICHAR, name.ToUpperInvariant());
      m_Writer.WriteLine(header);
    }

    private void writeFooter(string name)
    {
      var footer = string.Format("{0}{1}{0}", DCHAR, name.ToUpperInvariant());
      m_Writer.WriteLine(footer);
    }

    private bool readBeginTag(string name)
    {
      var header = m_Reader.ReadLine();
      if (string.IsNullOrWhiteSpace(name) ||
          header.Length<3 ||
          !header.StartsWith(ICHAR) ||
          !header.EndsWith(ICHAR)) return false;

      var body = header.Substring(1, header.Length-2);
      if (!string.Equals(body, name, StringComparison.InvariantCultureIgnoreCase)) return false;

      return true;
    }

    private string readBlock(string name)
    {
      var res = readBeginTag(name);
      if (!res) throw new MLException(string.Format("Error while reading begin tag {0}", name ?? "NULL"));

      var builder = new StringBuilder();
      var first = true;

      while (true)
      {
        var line = m_Reader.ReadLine();
        if (line==null)
          throw new MLException(string.Format("End tag {0} not found", name ?? "NULL"));

        if (isEndTag(line, name)) break;
        else
        {
          if (!first) builder.AppendLine();
          builder.Append(line);
          first = false;
        }
      }

      return builder.ToString();
    }

    private IEnumerable<string> readList(string name)
    {
      var res = readBeginTag(name);
      if (!res) throw new MLException(string.Format("Error while reading begin tag {0}", name ?? "NULL"));

      while (true)
      {
        var line = m_Reader.ReadLine();
        if (line==null)
          throw new MLException(string.Format("End tag {0} not found", name ?? "NULL"));

        if (isEndTag(line, name)) yield break;
        else yield return line;
      }
    }

    private bool isEndTag(string line, string name)
    {
      if (string.IsNullOrWhiteSpace(name) ||
          line.Length<3 ||
          !line.StartsWith(DCHAR) ||
          !line.EndsWith(DCHAR)) return false;

      var body = line.Substring(1, line.Length-2);
      if (!string.Equals(body, name, StringComparison.InvariantCultureIgnoreCase)) return false;

      return true;
    }

    #endregion
  }
}

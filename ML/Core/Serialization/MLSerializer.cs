using ML.Contracts;
using System;
using System.Collections.Generic;
using System.IO;

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

    public string Read(string name, string dft)
    {
      var res = readBeginTag(name);
      if (!res) throw new MLException(string.Format("Error while reading begin tag {0}", name ?? "NULL"));

      // TODO

      res = readEndTag(name);
      if (!res) throw new MLException(string.Format("Error while reading end tag {0}", name ?? "NULL"));

      return null;
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
        using (var reader = new StreamReader(ms))
        {
          var serializer = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
          serializer.Serialize(ms, v);
          ms.Position = 0;

          m_Writer.WriteLine(reader.ReadToEnd());
        }
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

    public void Write(string name, IEnumerable<Class> value)
    {
      write(name, value, v =>
      {
        foreach (var item in v)
          m_Writer.WriteLine("{0}:{1}", item.Value, item.Name);
      });
    }

    public void Write(string name, string value)
    {
      write(name, value, v =>
      {
        m_Writer.WriteLine(v);
      });
    }

    public void Write(string name, IDictionary<Class, double> value)
    {
      write(name, value, v =>
      {
        foreach (var kvp in v)
          m_Writer.WriteLine("{0}:{1:0.000000}", kvp.Key.Value, kvp.Value);
      });
    }

    public void Write(string name, IDictionary<Class, int> value)
    {
      write(name, value, v =>
      {
        foreach (var kvp in v)
          m_Writer.WriteLine("{0}:{1}", kvp.Key.Value, kvp.Value);
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


    private bool readEndTag(string name)
    {
      var footer = m_Reader.ReadLine();
      if (string.IsNullOrWhiteSpace(name) ||
          footer.Length<3 ||
          !footer.StartsWith(DCHAR) ||
          !footer.EndsWith(DCHAR)) return false;

      var body = footer.Substring(1, footer.Length-2);
      if (!string.Equals(body, name, StringComparison.InvariantCultureIgnoreCase)) return false;

      return true;
    }

    #endregion
  }
}

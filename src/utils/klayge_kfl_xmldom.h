/**
 * @file XMLDom.hpp
 * @author Minmin Gong
 *
 * @section DESCRIPTION
 *
 * This source file is part of KFL, a subproject of KlayGE
 * For the latest info, see http://www.klayge.org
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * You may alternatively use this source under the terms of
 * the KlayGE Proprietary License (KPL). You can obtained such a license
 * from http://www.klayge.org/licensing/.
 */

// KlayGE/KFL/include/KFL/XMLDom.hpp
// KlayGE/KFL/src/Kernel/XMLDom.cpp

#ifndef BUBBLEFS_UTILS_KLAYGE_KFL_XMLDOM_H_
#define BUBBLEFS_UTILS_KLAYGE_KFL_XMLDOM_H_

#include <iosfwd>
#include <memory>
#include <vector>
#include "utils/klayge_kfl_predeclare.h"
#include "utils/klayge_kfl_residentifier.h"

#include "boost/lexical_cast.hpp"
#include "rapidxml/rapidxml.hpp" // add -fdelayed-template-parsing to prevent compile error
#include "rapidxml/rapidxml_print.hpp"

namespace bubblefs {
namespace myklayge {

        enum XMLNodeType
        {
                XNT_Document,
                XNT_Element,
                XNT_Data,
                XNT_CData,
                XNT_Comment,
                XNT_Declaration,
                XNT_Doctype,
                XNT_PI
        };

        class XMLDocument
        {
        public:
                XMLDocument();

                XMLNodePtr Parse(ResIdentifierPtr const & source);
                void Print(std::ostream& os);

                XMLNodePtr CloneNode(XMLNodePtr const & node);

                XMLNodePtr AllocNode(XMLNodeType type, std::string_view name);
                XMLAttributePtr AllocAttribInt(std::string_view name, int32_t value);
                XMLAttributePtr AllocAttribUInt(std::string_view name, uint32_t value);
                XMLAttributePtr AllocAttribFloat(std::string_view name, float value);
                XMLAttributePtr AllocAttribString(std::string_view name, std::string_view value);

                void RootNode(XMLNodePtr const & new_node);

        private:
                std::shared_ptr<void> doc_;
                std::vector<char> xml_src_;

                XMLNodePtr root_;
        };

        class XMLNode
        {
                friend class XMLDocument;

        public:
                explicit XMLNode(void* node);
                XMLNode(void* doc, XMLNodeType type, std::string_view name);

                std::string const & Name() const;
                XMLNodeType Type() const;

                XMLNodePtr Parent() const;

                XMLAttributePtr FirstAttrib(std::string_view name) const;
                XMLAttributePtr LastAttrib(std::string_view name) const;
                XMLAttributePtr FirstAttrib() const;
                XMLAttributePtr LastAttrib() const;

                XMLAttributePtr Attrib(std::string_view name) const;

                bool TryConvertAttrib(std::string_view name, int32_t& val, int32_t default_val) const;
                bool TryConvertAttrib(std::string_view name, uint32_t& val, uint32_t default_val) const;
                bool TryConvertAttrib(std::string_view name, float& val, float default_val) const;

                int32_t AttribInt(std::string_view name, int32_t default_val) const;
                uint32_t AttribUInt(std::string_view name, uint32_t default_val) const;
                float AttribFloat(std::string_view name, float default_val) const;
                std::string AttribString(std::string_view name, std::string default_val) const;

                XMLNodePtr FirstNode(std::string_view name) const;
                XMLNodePtr LastNode(std::string_view name) const;
                XMLNodePtr FirstNode() const;
                XMLNodePtr LastNode() const;

                XMLNodePtr PrevSibling(std::string_view name) const;
                XMLNodePtr NextSibling(std::string_view name) const;
                XMLNodePtr PrevSibling() const;
                XMLNodePtr NextSibling() const;

                void InsertNode(XMLNodePtr const & location, XMLNodePtr const & new_node);
                void InsertAttrib(XMLAttributePtr const & location, XMLAttributePtr const & new_attr);
                void AppendNode(XMLNodePtr const & new_node);
                void AppendAttrib(XMLAttributePtr const & new_attr);

                void RemoveNode(XMLNodePtr const & node);
                void RemoveAttrib(XMLAttributePtr const & attr);

                bool TryConvert(int32_t& val) const;
                bool TryConvert(uint32_t& val) const;
                bool TryConvert(float& val) const;

                int32_t ValueInt() const;
                uint32_t ValueUInt() const;
                float ValueFloat() const;
                std::string ValueString() const;

        private:
                void* node_;
                std::string name_;

                std::vector<XMLNodePtr> children_;
                std::vector<XMLAttributePtr> attrs_;
        };

        class XMLAttribute
        {
                friend class XMLDocument;
                friend class XMLNode;

        public:
                explicit XMLAttribute(void* attr);
                XMLAttribute(void* doc, std::string_view name, std::string_view value);

                std::string const & Name() const;

                XMLAttributePtr NextAttrib(std::string_view name) const;
                XMLAttributePtr NextAttrib() const;

                bool TryConvert(int32_t& val) const;
                bool TryConvert(uint32_t& val) const;
                bool TryConvert(float& val) const;

                int32_t ValueInt() const;
                uint32_t ValueUInt() const;
                float ValueFloat() const;
                std::string const & ValueString() const;

        private:
                void* attr_;
                std::string name_;
                std::string value_;
        };
        
        XMLDocument::XMLDocument()
                : doc_(MakeSharedPtr<rapidxml::xml_document<>>())
        {
        }

        XMLNodePtr XMLDocument::Parse(ResIdentifierPtr const & source)
        {
                source->seekg(0, std::ios_base::end);
                int len = static_cast<int>(source->tellg());
                source->seekg(0, std::ios_base::beg);
                xml_src_.resize(len + 1, 0);
                source->read(&xml_src_[0], len);

                static_cast<rapidxml::xml_document<>*>(doc_.get())->parse<0>(&xml_src_[0]);
                root_ = MakeSharedPtr<XMLNode>(static_cast<rapidxml::xml_document<>*>(doc_.get())->first_node());

                return root_;
        }

        void XMLDocument::Print(std::ostream& os)
        {
                os << "<?xml version=\"1.0\"?>" << std::endl << std::endl;
                os << *static_cast<rapidxml::xml_document<>*>(doc_.get());
        }

        XMLNodePtr XMLDocument::CloneNode(XMLNodePtr const & node)
        {
                return MakeSharedPtr<XMLNode>(static_cast<rapidxml::xml_document<>*>(doc_.get())->clone_node(static_cast<rapidxml::xml_node<>*>(node->node_)));
        }

        XMLNodePtr XMLDocument::AllocNode(XMLNodeType type, std::string_view name)
        {
                return MakeSharedPtr<XMLNode>(doc_.get(), type, name);
        }
        
        XMLAttributePtr XMLDocument::AllocAttribInt(std::string_view name, int32_t value)
        {
                return this->AllocAttribString(name, boost::lexical_cast<std::string>(value));
        }

        XMLAttributePtr XMLDocument::AllocAttribUInt(std::string_view name, uint32_t value)
        {
                return this->AllocAttribString(name, boost::lexical_cast<std::string>(value));
        }

        XMLAttributePtr XMLDocument::AllocAttribFloat(std::string_view name, float value)
        {
                return this->AllocAttribString(name, boost::lexical_cast<std::string>(value));
        }

        XMLAttributePtr XMLDocument::AllocAttribString(std::string_view name, std::string_view value)
        {
                return MakeSharedPtr<XMLAttribute>(doc_.get(), name, value);
        }

        void XMLDocument::RootNode(XMLNodePtr const & new_node)
        {
                static_cast<rapidxml::xml_document<>*>(doc_.get())->remove_all_nodes();
                static_cast<rapidxml::xml_document<>*>(doc_.get())->append_node(static_cast<rapidxml::xml_node<>*>(new_node->node_));
                root_ = new_node;
        }


        XMLNode::XMLNode(void* node)
                : node_(node)
        {
                if (node_ != nullptr)
                {
                        name_ = std::string(static_cast<rapidxml::xml_node<>*>(node_)->name(),
                                static_cast<rapidxml::xml_node<>*>(node_)->name_size());
                }
        }

        XMLNode::XMLNode(void* doc, XMLNodeType type, std::string_view name)
                : name_(name)
        {
                rapidxml::node_type xtype;
                switch (type)
                {
                case XNT_Document:
                        xtype = rapidxml::node_document;
                        break;

                case XNT_Element:
                        xtype = rapidxml::node_element;
                        break;

                case XNT_Data:
                        xtype = rapidxml::node_data;
                        break;

                case XNT_CData:
                        xtype = rapidxml::node_cdata;
                        break;

                case XNT_Comment:
                        xtype = rapidxml::node_comment;
                        break;

                case XNT_Declaration:
                        xtype = rapidxml::node_declaration;
                        break;

                case XNT_Doctype:
                        xtype = rapidxml::node_doctype;
                        break;

                case XNT_PI:
                default:
                        xtype = rapidxml::node_pi;
                        break;
                }

                node_ = static_cast<rapidxml::xml_document<>*>(doc)->allocate_node(xtype, name.data(), nullptr, name.size());
        }

        std::string const & XMLNode::Name() const
        {
                return name_;
        }

        XMLNodeType XMLNode::Type() const
        {
                switch (static_cast<rapidxml::xml_node<>*>(node_)->type())
                {
                case rapidxml::node_document:
                        return XNT_Document;

                case rapidxml::node_element:
                        return XNT_Element;

                case rapidxml::node_data:
                        return XNT_Data;

                case rapidxml::node_cdata:
                        return XNT_CData;

                case rapidxml::node_comment:
                        return XNT_Comment;

                case rapidxml::node_declaration:
                        return XNT_Declaration;

                case rapidxml::node_doctype:
                        return XNT_Doctype;

                case rapidxml::node_pi:
                default:
                        return XNT_PI;
                }
        }

        XMLNodePtr XMLNode::Parent() const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->parent();
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLAttributePtr XMLNode::FirstAttrib(std::string_view name) const
        {
                rapidxml::xml_attribute<>* attr = static_cast<rapidxml::xml_node<>*>(node_)->first_attribute(name.data(), name.size());
                if (attr)
                {
                        return MakeSharedPtr<XMLAttribute>(attr);
                }
                else
                {
                        return XMLAttributePtr();
                }
        }
        
        XMLAttributePtr XMLNode::LastAttrib(std::string_view name) const
        {
                rapidxml::xml_attribute<>* attr = static_cast<rapidxml::xml_node<>*>(node_)->last_attribute(name.data(), name.size());
                if (attr)
                {
                        return MakeSharedPtr<XMLAttribute>(attr);
                }
                else
                {
                        return XMLAttributePtr();
                }
        }

        XMLAttributePtr XMLNode::FirstAttrib() const
        {
                rapidxml::xml_attribute<>* attr = static_cast<rapidxml::xml_node<>*>(node_)->first_attribute();
                if (attr)
                {
                        return MakeSharedPtr<XMLAttribute>(attr);
                }
                else
                {
                        return XMLAttributePtr();
                }
        }

        XMLAttributePtr XMLNode::LastAttrib() const
        {
                rapidxml::xml_attribute<>* attr = static_cast<rapidxml::xml_node<>*>(node_)->last_attribute();
                if (attr)
                {
                        return MakeSharedPtr<XMLAttribute>(attr);
                }
                else
                {
                        return XMLAttributePtr();
                }
        }

        XMLAttributePtr XMLNode::Attrib(std::string_view name) const
        {
                return this->FirstAttrib(name);
        }

        bool XMLNode::TryConvertAttrib(std::string_view name, int32_t& val, int32_t default_val) const
        {
                val = default_val;

                XMLAttributePtr attr = this->Attrib(name);
                return attr ? attr->TryConvert(val) : true;
        }

        bool XMLNode::TryConvertAttrib(std::string_view name, uint32_t& val, uint32_t default_val) const
        {
                val = default_val;

                XMLAttributePtr attr = this->Attrib(name);
                return attr ? attr->TryConvert(val) : true;
        }

        bool XMLNode::TryConvertAttrib(std::string_view name, float& val, float default_val) const
        {
                val = default_val;

                XMLAttributePtr attr = this->Attrib(name);
                return attr ? attr->TryConvert(val) : true;
        }

        int32_t XMLNode::AttribInt(std::string_view name, int32_t default_val) const
        {
                XMLAttributePtr attr = this->Attrib(name);
                return attr ? attr->ValueInt() : default_val;
        }

        uint32_t XMLNode::AttribUInt(std::string_view name, uint32_t default_val) const
        {
                XMLAttributePtr attr = this->Attrib(name);
                return attr ? attr->ValueUInt() : default_val;
        }

        float XMLNode::AttribFloat(std::string_view name, float default_val) const
        {
                XMLAttributePtr attr = this->Attrib(name);
                return attr ? attr->ValueFloat() : default_val;
        }

        std::string XMLNode::AttribString(std::string_view name, std::string default_val) const
        {
                XMLAttributePtr attr = this->Attrib(name);
                return attr ? attr->ValueString() : default_val;
        }

        XMLNodePtr XMLNode::FirstNode(std::string_view name) const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->first_node(name.data(), name.size());
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLNodePtr XMLNode::LastNode(std::string_view name) const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->last_node(name.data(), name.size());
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLNodePtr XMLNode::FirstNode() const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->first_node();
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLNodePtr XMLNode::LastNode() const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->last_node();
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLNodePtr XMLNode::PrevSibling(std::string_view name) const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->previous_sibling(name.data(), name.size());
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLNodePtr XMLNode::NextSibling(std::string_view name) const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->next_sibling(name.data(), name.size());
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLNodePtr XMLNode::PrevSibling() const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->previous_sibling();
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        XMLNodePtr XMLNode::NextSibling() const
        {
                rapidxml::xml_node<>* node = static_cast<rapidxml::xml_node<>*>(node_)->next_sibling();
                if (node)
                {
                        return MakeSharedPtr<XMLNode>(node);
                }
                else
                {
                        return XMLNodePtr();
                }
        }

        void XMLNode::InsertNode(XMLNodePtr const & location, XMLNodePtr const & new_node)
        {
                static_cast<rapidxml::xml_node<>*>(node_)->insert_node(static_cast<rapidxml::xml_node<>*>(location->node_),
                        static_cast<rapidxml::xml_node<>*>(new_node->node_));
                for (size_t i = 0; i < children_.size(); ++ i)
                {
                        if (children_[i]->node_ == location->node_)
                        {
                                children_.insert(children_.begin() + i, new_node);
                                break;
                        }
                }
        }

        void XMLNode::InsertAttrib(XMLAttributePtr const & location, XMLAttributePtr const & new_attr)
        {
                static_cast<rapidxml::xml_node<>*>(node_)->insert_attribute(static_cast<rapidxml::xml_attribute<>*>(location->attr_),
                        static_cast<rapidxml::xml_attribute<>*>(new_attr->attr_));
                for (size_t i = 0; i < attrs_.size(); ++ i)
                {
                        if (attrs_[i]->attr_ == location->attr_)
                        {
                                attrs_.insert(attrs_.begin() + i, new_attr);
                                break;
                        }
                }
        }

        void XMLNode::AppendNode(XMLNodePtr const & new_node)
        {
                static_cast<rapidxml::xml_node<>*>(node_)->append_node(static_cast<rapidxml::xml_node<>*>(new_node->node_));
                children_.push_back(new_node);
        }

        void XMLNode::AppendAttrib(XMLAttributePtr const & new_attr)
        {
                static_cast<rapidxml::xml_node<>*>(node_)->append_attribute(static_cast<rapidxml::xml_attribute<>*>(new_attr->attr_));
                attrs_.push_back(new_attr);
        }

        void XMLNode::RemoveNode(XMLNodePtr const & node)
        {
                static_cast<rapidxml::xml_node<>*>(node_)->remove_node(static_cast<rapidxml::xml_node<>*>(node->node_));
                for (size_t i = 0; i < children_.size(); ++ i)
                {
                        if (children_[i]->node_ == node->node_)
                        {
                                children_.erase(children_.begin() + i);
                                break;
                        }
                }
        }

        void XMLNode::RemoveAttrib(XMLAttributePtr const & attr)
        {
                static_cast<rapidxml::xml_node<>*>(node_)->remove_attribute(static_cast<rapidxml::xml_attribute<>*>(attr->attr_));
                for (size_t i = 0; i < attrs_.size(); ++ i)
                {
                        if (attrs_[i]->attr_ == attr->attr_)
                        {
                                attrs_.erase(attrs_.begin() + i);
                                break;
                        }
                }
        }

        bool XMLNode::TryConvert(int32_t& val) const
        {
                return boost::conversion::try_lexical_convert(this->ValueString(), val);
        }

        bool XMLNode::TryConvert(uint32_t& val) const
        {
                return boost::conversion::try_lexical_convert(this->ValueString(), val);
        }

        bool XMLNode::TryConvert(float& val) const
        {
                return boost::conversion::try_lexical_convert(this->ValueString(), val);
        }

        int32_t XMLNode::ValueInt() const
        {
                return boost::lexical_cast<int32_t>(this->ValueString());
        }

        uint32_t XMLNode::ValueUInt() const
        {
                return boost::lexical_cast<uint32_t>(this->ValueString());
        }

        float XMLNode::ValueFloat() const
        {
                return boost::lexical_cast<float>(this->ValueString());
        }

        std::string XMLNode::ValueString() const
        {
                return std::string(static_cast<rapidxml::xml_node<>*>(node_)->value(),
                        static_cast<rapidxml::xml_node<>*>(node_)->value_size());
        }


        XMLAttribute::XMLAttribute(void* attr)
                : attr_(attr)
        {
                if (attr_ != nullptr)
                {
                        auto const xml_attr = static_cast<rapidxml::xml_attribute<>*>(attr_);
                        name_ = std::string(xml_attr->name(), xml_attr->name_size());
                        value_ = std::string(xml_attr->value(), xml_attr->value_size());
                }
        }

        XMLAttribute::XMLAttribute(void* doc, std::string_view name, std::string_view value)
                : name_(name), value_(value)
        {
                attr_ = static_cast<rapidxml::xml_document<>*>(doc)->allocate_attribute(name.data(), value.data(), name.size(), value.size());
        }

        std::string const & XMLAttribute::Name() const
        {
                return name_;
        }

        XMLAttributePtr XMLAttribute::NextAttrib(std::string_view name) const
        {
                rapidxml::xml_attribute<>* attr = static_cast<rapidxml::xml_attribute<>*>(attr_)->next_attribute(name.data(), name.size());
                if (attr)
                {
                        return MakeSharedPtr<XMLAttribute>(attr);
                }
                else
                {
                        return XMLAttributePtr();
                }
        }

        XMLAttributePtr XMLAttribute::NextAttrib() const
        {
                rapidxml::xml_attribute<>* attr = static_cast<rapidxml::xml_attribute<>*>(attr_)->next_attribute();
                if (attr)
                {
                        return MakeSharedPtr<XMLAttribute>(attr);
                }
                else
                {
                        return XMLAttributePtr();
                }
        }

        bool XMLAttribute::TryConvert(int32_t& val) const
        {
                return boost::conversion::try_lexical_convert(value_, val);
        }

        bool XMLAttribute::TryConvert(uint32_t& val) const
        {
                return boost::conversion::try_lexical_convert(value_, val);
        }

        bool XMLAttribute::TryConvert(float& val) const
        {
                return boost::conversion::try_lexical_convert(value_, val);
        }

        int32_t XMLAttribute::ValueInt() const
        {
                return boost::lexical_cast<int32_t>(value_);
        }

        uint32_t XMLAttribute::ValueUInt() const
        {
                return boost::lexical_cast<uint32_t>(value_);
        }

        float XMLAttribute::ValueFloat() const
        {
                return boost::lexical_cast<float>(value_);
        }

        std::string const & XMLAttribute::ValueString() const
        {
                return value_;
        }
        
} // namespace myklayge
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_KLAYGE_KFL_XMLDOM_H_
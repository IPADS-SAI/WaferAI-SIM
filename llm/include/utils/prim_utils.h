#pragma once
#include "common/system.h"
#include "systemc.h"
#include <functional>
#include <memory>
#include <unordered_map>

#include "prims/base.h"
#include "prims/comp_prims.h"
#include "prims/gpu_prims.h"
#include "prims/norm_prims.h"
#include "utils/print_utils.h"

class PrimFactory {
public:
    using CreatorFunc = std::function<PrimBase *()>;

    static PrimFactory &getInstance() {
        static PrimFactory instance;
        return instance;
    }

    void registerPrim(const std::string &type, CreatorFunc creator) {
        creators_[type] = creator;
        type_to_id_[type] = next_id_;
        id_to_type_[next_id_] = type;

        next_id_++;
    }

    PrimBase *createPrim(const std::string &type, bool need_init = true) {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            PrimBase *prim = it->second();
            if (need_init)
                prim->prim_context = make_shared<PrimCoreContext>();

            g_prim_stash.push_back(prim);
            return prim;
        }

        LOG_ERROR(prim_utils.h) << "Unregistered primitive type " << type;
        return nullptr;
    }

    PrimBase *createPrim(int id, bool need_init = true) {
        auto it = id_to_type_.find(id);
        if (it != id_to_type_.end()) {
            return createPrim(it->second, need_init);
        }

        LOG_ERROR(prim_utils.h) << "Unregistered primitive ID " << id;
        return nullptr;
    }

    int getPrimId(const std::string &type) const {
        auto it = type_to_id_.find(type);
        if (it != type_to_id_.end()) {
            return it->second;
        }
        return -1;
    }

    const std::string &getPrimType(int id) const {
        static const std::string empty_string = "";
        auto it = id_to_type_.find(id);
        if (it != id_to_type_.end()) {
            return it->second;
        }
        return empty_string;
    }

private:
    std::unordered_map<std::string, CreatorFunc> creators_;
    std::unordered_map<std::string, int> type_to_id_;
    std::unordered_map<int, std::string> id_to_type_;
    int next_id_ = 1;

    PrimFactory() = default;
};

// 所有原语的注册函数
#define REGISTER_PRIM(prim_type)                                               \
    static bool registered_##prim_type = []() {                                \
        PrimFactory::getInstance().registerPrim(                               \
            prim_type().name, []() { return new prim_type(); });               \
        return true;                                                           \
    }();
